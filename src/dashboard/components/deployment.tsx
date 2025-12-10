"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Rocket, CheckCircle2, AlertCircle } from "lucide-react"
import { getMLflowExperiments, getMLflowRuns } from "@/lib/api"
import { DeploymentPipeline } from "./Pipeline"

const DEPLOYMENT_STEPS = [
  { id: "SELECT_MODEL", label: "Select Model" },
  { id: "BUILD_IMAGE", label: "Build Image" },
  { id: "APPLY_MANIFESTS", label: "Apply Config" },
  { id: "ROLLOUT_RESTART", label: "Restarting" },
  { id: "WAIT_ROLLOUT", label: "Verifying" },
];

interface BestModel {
  runId: string;
  name: string;
  f1: number;
  accuracy: number;
  experimentId: string;
}

export default function Deployment() {
  const [bestModel, setBestModel] = useState<BestModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [deployed, setDeployed] = useState(false);
  const [deployedRunId, setDeployedRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("IDLE");
  const [currentStep, setCurrentStep] = useState<string>("");
  const [statusMessage, setStatusMessage] = useState<string>("");

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/v1/deploy/status");
        const data = await res.json();

        if (data.status === "DEPLOYING") {
          setDeploying(true);
          setDeployed(false);
          setStatus("DEPLOYING");
          setCurrentStep(data.step || "SELECT_MODEL");
          setStatusMessage(data.message || "Initializing deployment...");
        } else if (data.status === "SUCCESS") {
          setDeploying(false);
          // Only successfully deployed IF the deployed run ID matches the current best model ID
          // OR if we don't know the deployed ID (legacy), we behave as before (true)
          // But here we want to fix the bug, so:
          if (data.deployed_run_id && bestModel && data.deployed_run_id !== bestModel.runId) {
            setDeployed(false);
          } else {
            setDeployed(true);
          }
          setDeployedRunId(data.deployed_run_id);
          setDeployedRunId(data.deployed_run_id);
          setStatus("SUCCESS");
          setCurrentStep("DONE");
          setStatusMessage(data.message || "Deployment successful");
        } else if (data.status === "FAILED") {
          setDeploying(false);
          setDeployed(false);
          setStatus("FAILED");
          setCurrentStep("FAILED");
          setStatusMessage(data.message || "Deployment failed");
        } else {
          setStatusMessage("");
        }
      } catch (e) {
        console.error("Polling error", e);
      }
    };

    checkStatus();

    const interval = setInterval(checkStatus, 1000); // Faster polling for smooth UI
    return () => clearInterval(interval);
  }, [bestModel]); // Add bestModel as dependency to re-eval when it changes



  useEffect(() => {
    async function findBestModel() {
      try {
        const expRes = await getMLflowExperiments();
        if (!expRes.experiments || expRes.experiments.length === 0) {
          setLoading(false);
          return;
        }

        let currentBest: BestModel | null = null;

        for (const exp of expRes.experiments) {
          try {
            const runsRes = await getMLflowRuns(exp.experiment_id);
            if (runsRes.runs) {
              for (const run of runsRes.runs) {
                const metrics: Record<string, number> = {};
                run.data.metrics.forEach(m => {
                  metrics[m.key] = m.value;
                });

                const f1 = metrics.test_f1 || metrics.f1_score || 0;
                const accuracy = metrics.test_accuracy || metrics.accuracy || 0;

                if (f1 > 0 || accuracy > 0) {
                  if (!currentBest || f1 > currentBest.f1 || (f1 === currentBest.f1 && accuracy > currentBest.accuracy)) {
                    const tags: Record<string, string> = {};
                    run.data.tags.forEach(t => {
                      tags[t.key] = t.value;
                    });

                    currentBest = {
                      runId: run.info.run_id,
                      name: tags.model_type || tags.model || run.info.run_name || "Unknown Model",
                      f1,
                      accuracy,
                      experimentId: exp.experiment_id,
                    };
                  }
                }
              }
            }
          } catch (err) {
            console.error(err);
          }
        }

        setBestModel(currentBest);
        setLoading(false);
      } catch (error) {
        console.error("Error finding best model:", error);
        setLoading(false);
      }
    }

    findBestModel();
    const interval = setInterval(findBestModel, 5000);
    return () => clearInterval(interval);
  }, []);


  const handleDeploy = async () => {
    if (!bestModel) return;
    setDeploying(true);
    setDeployed(false);
    setStatusMessage("Queueing deployment job...");

    try {
      const response = await fetch("http://localhost:8000/api/v1/deploy", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          experiment_id: bestModel.experimentId
        })
      });


      if (!response.ok) {
        throw new Error("Deployment request failed");
      }

      const result = await response.json();
      console.log(result.message);

    } catch (error) {
      console.error("Error deploying model:", error);
      setDeploying(false); // Reset on error
      setStatusMessage("Failed to start deployment");
    }
  };


  if (loading) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Rocket className="h-4 w-4 text-primary" />
            Deployment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Finding best model...</p>
        </CardContent>
      </Card>
    );
  }

  if (!bestModel) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-sm flex items-center gap-2">
            <Rocket className="h-4 w-4 text-primary" />
            Deployment
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-yellow-500">
            <AlertCircle className="h-4 w-4" />
            <p className="text-sm">No trained models available for deployment.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Determine if we need to show the deploy button
  // If not deployed, OR if deployed but the runId doesn't match the best model's runId
  const needDeployment = !deployed || (deployedRunId && deployedRunId !== bestModel.runId);

  // NOTE: deployed state variable is now a bit ambiguous because we update it in checkStatus based on runId check.
  // However, checkStatus is efficient. 
  // Let's refine the "deployed" state in checkStatus actually to be the source of truth.
  // Above I added the check in checkStatus. So 'deployed' is already "is successfully deployed version match".

  // Wait, if I just start the app, deployed is false. checkStatus runs.
  // if backend says SUCCESS and id matches, deployed -> true.
  // if backend says SUCCESS and id MISMATCH, deployed -> false.

  // What if deploying? 
  // if backend says DEPLOYING, deployed -> false, deploying -> true.

  return (
    <Card className="border-border bg-card border-l-4 border-l-primary">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <Rocket className="h-5 w-5 text-primary" />
          Deploy Best Model
        </CardTitle>
        <CardDescription>
          Ready to deploy <strong>{bestModel.name}</strong> to production.
        </CardDescription>
      </CardHeader>
      <CardContent className="pb-4">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-muted/30 p-3 rounded-md">
            <p className="text-xs text-muted-foreground uppercase font-semibold">F1 Score</p>
            <p className="text-2xl font-bold text-foreground">{bestModel.f1.toFixed(3)}</p>
          </div>
          <div className="bg-muted/30 p-3 rounded-md">
            <p className="text-xs text-muted-foreground uppercase font-semibold">Accuracy</p>
            <p className="text-2xl font-bold text-foreground">{bestModel.accuracy.toFixed(3)}</p>
          </div>
        </div>
        <p className="text-xs text-muted-foreground mb-4">
          Run ID: <span className="font-mono">{bestModel.runId}</span>
        </p>

        {statusMessage && (deploying || status === "FAILED") && (
          <div className="bg-accent/50 p-3 rounded-md mb-2 flex items-center gap-2">
            {deploying && <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin shrink-0" />}
            <p className="text-sm font-medium">{statusMessage}</p>
          </div>
        )}

        {(deploying || status === "SUCCESS" || status === "FAILED") && (
          <DeploymentPipeline currentStep={currentStep} steps={DEPLOYMENT_STEPS} />
        )}

      </CardContent>
      <CardFooter>
        {deployed ? (
          <Button
            className="w-full !bg-green-600 hover:!bg-green-700 !text-white border-green-700"
            onClick={handleDeploy}
            disabled={deploying}
          >

            {deploying ? (
              <>
                <Rocket className="mr-2 h-4 w-4 animate-bounce" />
                Redeploying...
              </>
            ) : (
              <>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Deployed Successfully (Click to Redeploy)
              </>
            )}
          </Button>
        ) : (

          <Button
            className="w-full"
            onClick={handleDeploy}
            disabled={deploying}
          >
            {deploying ? (
              <>
                Deployment in Progress...
              </>
            ) : (
              <>
                <Rocket className="mr-2 h-4 w-4" />
                Deploy to Production
              </>
            )}
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
