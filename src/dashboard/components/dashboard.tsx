"use client"

import { useEffect, useState, useRef } from "react"
import { useRouter, useSearchParams } from "next/navigation"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import StatusCards from "./status_card"
import ModelLeaderboard from "./model_leaderboard"
import MetricsCharts from "./metrics"
import TrainingEvolution from "./training_evolution"
import PipelineVisualizer from "./pipeline-visualizer"
import ModelRegistry from "./model_registry"
import Deployment from "./deployment"
import { useDashboardData } from "../hooks/use-dashboard-data"
import { getJobs, type JobSummary } from "../lib/api"
import { ChevronDown, Search } from "lucide-react"

export default function Dashboard() {
  const searchParams = useSearchParams()
  const router = useRouter()

  const initialRequestId = searchParams?.get("requestId") || ""
  const [requestId, setRequestId] = useState<string | null>(initialRequestId || null)
  const [requestInput, setRequestInput] = useState(initialRequestId)

  const [availableJobs, setAvailableJobs] = useState<JobSummary[]>([])
  const [showJobDropdown, setShowJobDropdown] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const fromUrl = searchParams?.get("requestId") || ""
    setRequestId(fromUrl || null)
    setRequestInput(fromUrl)
  }, [searchParams])

  useEffect(() => {
    async function fetchJobs() {
      try {
        const res = await getJobs();
        if (res.jobs) {
          setAvailableJobs(res.jobs);
        }
      } catch (error) {
        console.error("Failed to fetch jobs:", error);
      }
    }
    fetchJobs();

    // Close dropdown when clicking outside
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowJobDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const { health, jobStatus, loadingJob, jobError } = useDashboardData({ requestId })

  const selectedJob = availableJobs.find(j => j.request_id === requestId);
  const experimentName = selectedJob?.experiment_name;

  const handleTrackRequest = (id?: string) => {
    const nextId = (id || requestInput).trim()
    const normalized = nextId.length ? nextId : null
    setRequestId(normalized)
    setRequestInput(normalized || "")
    const url = normalized ? `?requestId=${encodeURIComponent(normalized)}` : ""
    router.replace(url, { scroll: false })
    setShowJobDropdown(false)
  }

  return (
    <div className="space-y-6">
      {/* Job tracker */}
      <Card className="border-border bg-card overflow-visible">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Track a training job</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3 md:flex-row md:items-center relative" ref={dropdownRef}>
          <div className="relative w-full md:max-w-sm">
            <input
              value={requestInput}
              onChange={(e) => setRequestInput(e.target.value)}
              onFocus={() => setShowJobDropdown(true)}
              placeholder="Paste request id or select from list"
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm pr-8"
            />
            <ChevronDown className="absolute right-2 top-2.5 h-4 w-4 text-muted-foreground pointer-events-none" />

            {showJobDropdown && availableJobs.length > 0 && (
              <div className="absolute top-full left-0 w-full mt-1 bg-popover border border-border rounded-md shadow-md z-50 max-h-60 overflow-y-auto">
                {availableJobs.map((job) => (
                  <div
                    key={job.request_id}
                    className="px-3 py-2 hover:bg-accent cursor-pointer text-sm"
                    onClick={() => handleTrackRequest(job.request_id)}
                  >
                    <div className="font-medium truncate">{job.dataset_name}</div>
                    {job.experiment_name && (
                      <div className="text-xs text-muted-foreground truncate">Exp: {job.experiment_name}</div>
                    )}
                    <div className="text-xs text-muted-foreground flex justify-between">
                      <span className="truncate max-w-[150px]">{job.request_id}</span>
                      <span>{new Date(job.created_at).toLocaleDateString()}</span>
                    </div>
                    <div className={`text-xs mt-1 ${
                      job.status === "COMPLETED" ? "text-green-500" :
                      job.status === "FAILED" ? "text-red-500" : "text-yellow-500"
                    }`}>
                      {job.status}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <Button onClick={() => handleTrackRequest()} size="sm" className="md:w-auto w-full">
            Track
          </Button>
          {jobError && <span className="text-xs text-red-400">{jobError}</span>}
        </CardContent>
      </Card>

      {/* Status Cards */}
      <StatusCards health={health} jobStatus={jobStatus} loadingJob={loadingJob} requestId={requestId} jobError={jobError} />

      {/* Pipeline Visualizer */}
      <PipelineVisualizer jobStatus={jobStatus} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Leaderboard */}
        <div className="lg:col-span-1">
          <ModelLeaderboard experimentName={experimentName} />
        </div>

        {/* Metrics Charts */}
        <div className="lg:col-span-2">
          <MetricsCharts experimentName={experimentName} />
        </div>
      </div>

      {/* Training Evolution */}
      <TrainingEvolution />

      {/* Registry and Deployment Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Registry */}
        <div className="lg:col-span-2">
          <ModelRegistry />
        </div>

        {/* Deployment */}
        <div className="lg:col-span-1">
          <Deployment />
        </div>
      </div>
    </div>
  )
}
