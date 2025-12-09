"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { getMLflowExperiments, getMLflowRuns, getMLflowMetricHistory, type MLflowMetric } from "@/lib/api"
import { Activity } from "lucide-react"

interface MetricPoint {
  step: number;
  train_loss?: number;
  test_loss?: number;
}

export default function TrainingEvolution() {
  const [data, setData] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [runName, setRunName] = useState<string>("");

  useEffect(() => {
    async function fetchData() {
      try {
        const expRes = await getMLflowExperiments();
        if (!expRes.experiments || expRes.experiments.length === 0) {
          setLoading(false);
          return;
        }
        let latestRun = null;
        let latestTime = 0;

        for (const exp of expRes.experiments) {
          try {
            const runsRes = await getMLflowRuns(exp.experiment_id);
            if (runsRes.runs) {
              for (const run of runsRes.runs) {
                const tags: Record<string, string> = {};
                run.data.tags.forEach(t => {
                  tags[t.key] = t.value;
                });

                if (tags.model_type === "Sequential" || tags.model === "Sequential") {
                  const startTime = run.info.start_time || 0;
                  if (startTime > latestTime) {
                    latestTime = startTime;
                    latestRun = run;
                  }
                }
              }
            }
          } catch (err) {
            console.error(err);
          }
        }

        if (latestRun) {
          setRunName(latestRun.info.run_name);
          const runId = latestRun.info.run_id;

          // Fetch metric history
          const [trainLossRes, testLossRes] = await Promise.all([
            getMLflowMetricHistory(runId, "train_loss").catch(() => ({ metrics: [] })),
            getMLflowMetricHistory(runId, "test_loss").catch(() => ({ metrics: [] })),
          ]);

          const history: Record<number, MetricPoint> = {};

          trainLossRes.metrics?.forEach((m: MLflowMetric) => {
            const step = m.step || 0;
            if (!history[step]) history[step] = { step };
            history[step].train_loss = m.value;
          });

          testLossRes.metrics?.forEach((m: MLflowMetric) => {
            const step = m.step || 0;
            if (!history[step]) history[step] = { step };
            history[step].test_loss = m.value;
          });

          const sortedData = Object.values(history).sort((a, b) => a.step - b.step);
          setData(sortedData);
        }

        setLoading(false);
      } catch (error) {
        console.error("Error fetching training evolution:", error);
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card className="border-border bg-card h-[400px]">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" />
            Training Evolution
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 flex items-center justify-center h-full">
          <p className="text-muted-foreground">Loading training history...</p>
        </CardContent>
      </Card>
    );
  }

  if (data.length === 0) {
    return null; // Don't show if no data (e.g. no Sequential models yet)
  }

  return (
    <Card className="border-border bg-card h-[400px] flex flex-col">
      <CardHeader className="pb-4 border-b border-border">
        <CardTitle className="text-sm flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Training Evolution
        </CardTitle>
        <CardDescription>Loss over epochs for {runName}</CardDescription>
      </CardHeader>
      <CardContent className="p-6 flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="step"
              stroke="#888888"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              label={{ value: 'Epoch', position: 'insideBottomRight', offset: -5 }}
            />
            <YAxis
              stroke="#888888"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value.toFixed(3)}`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
              itemStyle={{ color: '#fff' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="train_loss"
              stroke="#3b82f6"
              name="Train Loss"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="test_loss"
              stroke="#ef4444"
              name="Test Loss"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
