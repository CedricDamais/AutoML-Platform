"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Trophy, Zap } from "lucide-react"
import { getMLflowExperiments, getMLflowRuns } from "@/lib/api"

interface ModelMetrics {
  rank: number;
  algorithm: string;
  f1?: number;
  accuracy?: number;
  latency?: number;
  status: string;
  runId: string;
}

const statusConfig: Record<string, { bg: string; text: string }> = {
  Completed: { bg: "bg-emerald-500/20", text: "text-emerald-300" },
  Training: { bg: "bg-blue-500/20", text: "text-blue-300" },
  Failed: { bg: "bg-red-500/20", text: "text-red-300" },
}

interface ModelLeaderboardProps {
  experimentName?: string;
}

export default function ModelLeaderboard({ experimentName }: ModelLeaderboardProps) {
  const [modelData, setModelData] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!experimentName) {
      setModelData([]);
      setLoading(false);
      return;
    }

    setLoading(true);

    async function fetchMLflowData() {
      try {
        const expRes = await getMLflowExperiments();
        if (!expRes.experiments || expRes.experiments.length === 0) {
          setModelData([]);
          setLoading(false);
          return;
        }

        const targetExperiment = expRes.experiments.find(e => e.name === experimentName);

        if (!targetExperiment) {
          setModelData([]);
          setLoading(false);
          return;
        }

        const allRuns: Omit<ModelMetrics, "rank">[] = [];
        try {
          const runsRes = await getMLflowRuns(targetExperiment.experiment_id);
          if (runsRes.runs) {
            for (const run of runsRes.runs) {
              const metrics: Record<string, number> = {};
              run.data.metrics.forEach(m => {
                metrics[m.key] = m.value;
              });

              const tags: Record<string, string> = {};
              run.data.tags.forEach(t => {
                tags[t.key] = t.value;
              });

              const algorithm = tags.model_type || tags.model || "Unknown";
              const status = run.info.status === "FINISHED" ? "Completed" :
                              run.info.status === "RUNNING" ? "Training" :
                              run.info.status === "FAILED" ? "Failed" : "Unknown";

              allRuns.push({
                algorithm,
                f1: metrics.test_f1 || metrics.f1_score,
                accuracy: metrics.test_accuracy || metrics.accuracy,
                latency: metrics.inference_time || metrics.latency,
                status,
                runId: run.info.run_id,
              });
            }
          }
        } catch (err) {
          console.error(`Error fetching runs for experiment ${targetExperiment.experiment_id}:`, err);
        }

        // Sort by f1 score or accuracy (descending)
        allRuns.sort((a, b) => {
          const scoreA = a.f1 || a.accuracy || 0;
          const scoreB = b.f1 || b.accuracy || 0;
          return scoreB - scoreA;
        });

        // Add ranks
        const rankedData = allRuns.map((run, idx) => ({
          ...run,
          rank: idx + 1,
        }));

        setModelData(rankedData);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching MLflow data:", error);
        setLoading(false);
      }
    }

    fetchMLflowData();
    const interval = setInterval(fetchMLflowData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [experimentName]);

  if (!experimentName) {
    return (
      <Card className="border-border bg-card h-full flex flex-col">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm flex items-center gap-2">
            <Trophy className="h-4 w-4 text-yellow-400" />
            Model Leaderboard
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 flex items-center justify-center h-full">
          <p className="text-muted-foreground text-center">Select an experiment to view the leaderboard</p>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card className="border-border bg-card h-full flex flex-col">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm flex items-center gap-2">
            <Trophy className="h-4 w-4 text-yellow-400" />
            Model Leaderboard
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <p className="text-muted-foreground">Loading MLflow data...</p>
        </CardContent>
      </Card>
    );
  }

  if (modelData.length === 0) {
    return (
      <Card className="border-border bg-card h-full flex flex-col">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm flex items-center gap-2">
            <Trophy className="h-4 w-4 text-yellow-400" />
            Model Leaderboard
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4">
          <p className="text-muted-foreground">No trained models found. Train some models to see them here.</p>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card className="border-border bg-card h-full flex flex-col">
      <CardHeader className="pb-4 border-b border-border">
        <CardTitle className="text-sm flex items-center gap-2">
          <Trophy className="h-4 w-4 text-yellow-400" />
          Model Leaderboard
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto p-0">
        <table className="w-full high-density-table">
          <thead className="sticky top-0 bg-card border-b border-border">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-muted-foreground">Rank</th>
              <th className="px-3 py-2 text-left font-semibold text-muted-foreground">Algorithm</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">F1 Score</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">Accuracy</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">Latency</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">Status</th>
            </tr>
          </thead>
          <tbody>
            {modelData.map((row, idx) => {
              const isTopRow = idx === 0
              const statusConf = statusConfig[row.status as keyof typeof statusConfig] || { bg: "bg-gray-500/20", text: "text-gray-300" }
              return (
                <tr
                  key={row.runId}
                  className={`border-b border-border hover:bg-muted/50 transition-colors ${
                    isTopRow ? "table-row-highlight" : ""
                  }`}
                >
                  <td className="px-3 py-2.5 text-foreground font-semibold">{row.rank}</td>
                  <td className="px-3 py-2.5 text-foreground">{row.algorithm}</td>
                  <td className="px-3 py-2.5 text-right monospace-text font-medium">
                    {row.f1 !== undefined ? row.f1.toFixed(3) : <span className="text-muted-foreground">N/A</span>}
                  </td>
                  <td className="px-3 py-2.5 text-right monospace-text text-muted-foreground">
                    {row.accuracy !== undefined ? row.accuracy.toFixed(3) : <span className="text-muted-foreground">N/A</span>}
                  </td>
                  <td className="px-3 py-2.5 text-right monospace-text text-muted-foreground">
                    {row.latency !== undefined ? `${row.latency.toFixed(2)}ms` : <span className="text-muted-foreground">N/A</span>}
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <span className={`status-badge ${statusConf.bg} ${statusConf.text}`}>
                      {row.status === "Training" && <Zap className="h-3 w-3" />}
                      {row.status}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}
