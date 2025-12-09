"use client"

import { useEffect, useState } from "react"

import type { HealthResponse, JobStatusResponse } from "../lib/api"
import { getHealth, getJobStatus } from "../lib/api"

interface UseDashboardDataOptions {
  requestId?: string | null
  pollIntervalMs?: number
}

export function useDashboardData({ requestId, pollIntervalMs = 2000 }: UseDashboardDataOptions) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)

  const [jobStatus, setJobStatus] = useState<JobStatusResponse | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)
  const [loadingJob, setLoadingJob] = useState(false)

  // Fetch health once on mount
  useEffect(() => {
    let cancelled = false

    getHealth()
      .then((res) => {
        if (!cancelled) setHealth(res)
      })
      .catch((err: Error) => {
        if (!cancelled) setHealthError(err.message)
      })

    return () => {
      cancelled = true
    }
  }, [])

  // Poll job status when a request id is present
  useEffect(() => {
    if (!requestId) {
      setJobStatus(null)
      return undefined
    }

    let cancelled = false
    let timer: NodeJS.Timeout | null = null

    const fetchJob = async () => {
      setLoadingJob(true)
      try {
        const status = await getJobStatus(requestId)
        if (!cancelled) {
          setJobStatus(status)
          setJobError(null)

          // Log status updates for debugging
          console.log(`[Job ${requestId}] Status:`, status.status, status.message)
        }
      } catch (err) {
        if (!cancelled) setJobError((err as Error).message)
      } finally {
        if (!cancelled) setLoadingJob(false)
      }
    }

    // Initial fetch
    fetchJob()

    // Dynamic polling: faster for active jobs, slower for completed/failed
    const getDynamicInterval = () => {
      const status = jobStatus?.status?.toUpperCase()
      if (status === "COMPLETED" || status === "FAILED") {
        return 10000 // 10 seconds for finished jobs
      }
      return pollIntervalMs // 2 seconds for active jobs
    }

    // Polling loop with dynamic interval
    const scheduleNextPoll = () => {
      if (cancelled) return
      timer = setTimeout(() => {
        fetchJob().then(scheduleNextPoll)
      }, getDynamicInterval())
    }

    scheduleNextPoll()

    return () => {
      cancelled = true
      if (timer) clearInterval(timer)
    }
  }, [requestId, pollIntervalMs])

  return {
    health,
    healthError,
    jobStatus,
    jobError,
    loadingJob,
  }
}
