"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from "recharts"
import { getMLflowExperiments, getMLflowRuns } from "@/lib/api"

interface ModelMetricData {
  name: string;
  f1: number;
  accuracy: number;
  latency: number;
}

interface MetricsChartsProps {
  experimentName?: string;
}

export default function MetricsCharts({ experimentName }: MetricsChartsProps) {
  const [data, setData] = useState<ModelMetricData[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!experimentName) {
      setData([]);
      setLoading(false);
      return;
    }

    setLoading(true);

    async function fetchData() {
      try {
        const expRes = await getMLflowExperiments();
        if (!expRes.experiments || expRes.experiments.length === 0) {
          setData([]);
          setLoading(false);
          return;
        }

        const targetExperiment = expRes.experiments.find(e => e.name === experimentName);

        if (!targetExperiment) {
          setData([]);
          setLoading(false);
          return;
        }

        const allRuns: ModelMetricData[] = [];
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

              const name = tags.model_type || tags.model || run.info.run_name || "Unknown";

              // Only include runs that have at least one metric
              if (metrics.test_f1 || metrics.f1_score || metrics.test_accuracy || metrics.accuracy) {
                allRuns.push({
                  name,
                  f1: metrics.test_f1 || metrics.f1_score || 0,
                  accuracy: metrics.test_accuracy || metrics.accuracy || 0,
                  latency: metrics.inference_time || metrics.latency || 0,
                });
              }
            }
          }
        } catch (err) {
          console.error(err);
        }

        // Sort by F1 and take top 10
        allRuns.sort((a, b) => b.f1 - a.f1);
        setData(allRuns.slice(0, 10));
        setLoading(false);
      } catch (error) {
        console.error("Error fetching metrics:", error);
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [experimentName]);

  if (!experimentName) {
    return (
      <Card className="border-border bg-card">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm">Real-time Metrics</CardTitle>
        </CardHeader>
        <CardContent className="p-6 flex items-center justify-center">
          <p className="text-muted-foreground">Select an experiment to view metrics</p>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card className="border-border bg-card">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm">Real-time Metrics</CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <p className="text-muted-foreground">Loading metrics...</p>
        </CardContent>
      </Card>
    );
  }

  if (data.length === 0) {
    return (
      <Card className="border-border bg-card">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm">Real-time Metrics</CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <p className="text-muted-foreground">No metrics available yet.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-4 border-b border-border">
        <CardTitle className="text-sm">Real-time Metrics</CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <div className="space-y-6">
          {/* Model Comparison Chart */}
          <div>
            <h4 className="text-xs font-semibold text-muted-foreground mb-3">Top Models Performance (F1 Score)</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={data} layout="vertical" margin={{ top: 0, right: 30, left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
                <XAxis type="number" domain={[0, 1]} stroke="oklch(0.65 0 0)" style={{ fontSize: "12px" }} />
                <YAxis type="category" dataKey="name" stroke="oklch(0.65 0 0)" style={{ fontSize: "11px" }} width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "oklch(0.16 0 0)",
                    border: "1px solid oklch(0.25 0 0)",
                    borderRadius: "4px",
                  }}
                  labelStyle={{ color: "oklch(0.95 0 0)" }}
                />
                <Bar dataKey="f1" fill="oklch(0.53 0.28 264)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Accuracy vs Latency Scatter Plot */}
          {data.some(d => d.latency > 0) && (
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground mb-3">Accuracy vs Latency (ms)</h4>
              <ResponsiveContainer width="100%" height={200}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0 0)" />
                  <XAxis type="number" dataKey="latency" name="Latency" unit="ms" stroke="oklch(0.65 0 0)" style={{ fontSize: "12px" }} />
                  <YAxis type="number" dataKey="accuracy" name="Accuracy" domain={[0, 1]} stroke="oklch(0.65 0 0)" style={{ fontSize: "12px" }} />
                  <ZAxis type="category" dataKey="name" name="Model" />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{
                      backgroundColor: "oklch(0.16 0 0)",
                      border: "1px solid oklch(0.25 0 0)",
                      borderRadius: "4px",
                    }}
                    labelStyle={{ color: "oklch(0.95 0 0)" }}
                  />
                  <Scatter name="Models" data={data} fill="oklch(0.7 0.2 160)" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
