"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Activity, Clock3, Database, MessageSquare } from "lucide-react"

import type { HealthResponse, JobStatusResponse } from "./lib/api"

interface StatusCardsProps {
  health?: HealthResponse | null
  jobStatus?: JobStatusResponse | null
  loadingJob?: boolean
  requestId?: string | null
  jobError?: string | null
}

export default function StatusCards({ health, jobStatus, loadingJob, requestId, jobError }: StatusCardsProps) {
  const isApiOnline = health?.status?.toLowerCase().includes("up")
  const jobMessage = jobStatus?.message || jobError || "No message yet"
  const lastUpdated = jobStatus?.created_at ? new Date(jobStatus.created_at).toLocaleString() : "—"

  const statusCards = [
    {
      title: "API Health",
      value: isApiOnline ? "Online" : health?.status ?? "Checking...",
      icon: Activity,
      color: isApiOnline ? "bg-emerald-500/20 text-emerald-300" : "bg-amber-500/20 text-amber-300",
      loading: !health,
    },
    {
      title: "Job Status",
      value: jobStatus?.status || (requestId ? "Fetching..." : "Not tracking"),
      icon: Clock3,
      color: jobStatus?.status === "FAILED" ? "bg-red-500/20 text-red-300" : "bg-blue-500/20 text-blue-300",
      loading: loadingJob && !!requestId,
      helper: requestId ? `Request ${requestId}` : undefined,
    },
    {
      title: "Dataset",
      value: jobStatus?.dataset_name || "—",
      icon: Database,
      color: "bg-purple-500/20 text-purple-300",
      loading: loadingJob && !!requestId,
    },
    {
      title: "Last Update",
      value: lastUpdated,
      icon: MessageSquare,
      color: "bg-slate-500/20 text-slate-200",
      loading: loadingJob && !!requestId,
      helper: jobMessage,
    },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
      {statusCards.map((card, idx) => {
        const Icon = card.icon
        return (
          <Card key={idx} className="border-border bg-card">
            <CardHeader className="pb-3 space-y-1">
              <CardTitle className="text-xs font-medium text-muted-foreground flex items-center gap-2">
                <Icon className="h-4 w-4" />
                {card.title}
              </CardTitle>
              {card.helper && <p className="text-[11px] text-muted-foreground/80">{card.helper}</p>}
            </CardHeader>
            <CardContent>
              <div className="flex items-end justify-between">
                <div className={`text-2xl font-bold ${card.color.split(" ")[1]}`}>
                  {card.loading ? <span className="animate-pulse-ring inline-block">●</span> : card.value}
                </div>
                <div className={`p-3 rounded-lg ${card.color}`}>
                  <Icon className="h-5 w-5" />
                </div>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
