"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Rocket, CheckCircle2, AlertCircle } from "lucide-react"
import { getMLflowExperiments, getMLflowRuns } from "@/lib/api"

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
  }, []);

  const handleDeploy = () => {
    if (!bestModel) return;
    setDeploying(true);

    // Simulate deployment delay
    setTimeout(() => {
      setDeploying(false);
      setDeployed(true);
    }, 2000);
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
        <p className="text-xs text-muted-foreground">
          Run ID: <span className="font-mono">{bestModel.runId}</span>
        </p>
      </CardContent>
      <CardFooter>
        {deployed ? (
          <Button className="w-full bg-green-600 hover:bg-green-700 text-white" disabled>
            <CheckCircle2 className="mr-2 h-4 w-4" />
            Deployed Successfully
          </Button>
        ) : (
          <Button
            className="w-full"
            onClick={handleDeploy}
            disabled={deploying}
          >
            {deploying ? (
              <>
                <Rocket className="mr-2 h-4 w-4 animate-bounce" />
                Deploying...
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
