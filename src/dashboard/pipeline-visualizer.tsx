"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2, Loader2, Circle, Sparkles } from "lucide-react"

import type { JobStatusResponse } from "./lib/api"

const steps = [
	{ key: "UPLOAD", label: "Upload" },
	{ key: "BUILDING", label: "Building" },
	{ key: "TRAINING", label: "Training" },
	{ key: "EVALUATION", label: "Evaluation" },
	{ key: "READY", label: "Ready" },
]

function resolveActiveIndex(status?: string) {
	if (!status) return -1
	const normalized = status.toUpperCase()

	// Map various status values to pipeline steps
	if (normalized.includes("UPLOAD")) return 0
	// QUEUED implies upload is done, waiting for build -> Active step is Building
	if (normalized === "QUEUED" || normalized.includes("BUILD")) return 1
	if (normalized.includes("TRAIN")) return 2
	if (normalized.includes("EVALUAT")) return 3
	if (normalized.includes("COMPLETE") || normalized.includes("READY") || normalized.includes("DEPLOY")) return 4

	const idx = steps.findIndex((s) => s.key === normalized)
	return idx >= 0 ? idx : -1
}

interface PipelineVisualizerProps {
	jobStatus?: JobStatusResponse | null
}

export default function PipelineVisualizer({ jobStatus }: PipelineVisualizerProps) {
	const activeIdx = resolveActiveIndex(jobStatus?.status)

	return (
		<Card className="border-border bg-card">
			<CardHeader className="pb-4 border-b border-border">
				<CardTitle className="text-sm flex items-center gap-2">
					<Sparkles className="h-4 w-4 text-primary" />
					Pipeline Status
				</CardTitle>
			</CardHeader>
			<CardContent className="p-6">
				<div className="flex flex-col gap-6">
					<div className="grid grid-cols-5 gap-4">
						{steps.map((step, idx) => {
							const isActive = activeIdx === idx
							const isDone = activeIdx > idx

							// Choose icon based on state
							let Icon = Circle
							if (isDone) {
								Icon = CheckCircle2
							} else if (isActive) {
								Icon = Loader2
							}

							return (
								<div key={step.key} className="flex flex-col items-center text-center">
									<div
										className={`flex items-center justify-center h-12 w-12 rounded-full border transition-all duration-300 ${
											isDone
												? "border-emerald-500/60 bg-emerald-500/10 text-emerald-400"
												: isActive
													? "border-blue-500/60 bg-blue-500/10 text-blue-400 animate-bounce"
													: "border-border text-muted-foreground"
										}`}
									>
										<Icon className={`h-5 w-5 ${isActive ? "animate-spin" : ""}`} />
									</div>
									<p className={`mt-2 text-xs font-medium transition-colors ${
										isDone ? "text-emerald-400" : isActive ? "text-blue-400" : "text-muted-foreground"
									}`}>
										{step.label}
									</p>
								</div>
							)
						})}
					</div>

					<div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
						<div className="flex justify-between items-center mb-2">
							<p className="font-semibold text-foreground">Current status</p>
							<span className="text-xs text-muted-foreground">{jobStatus?.dataset_name || "—"}</span>
						</div>
						<p className="text-foreground">
							{jobStatus?.status ? jobStatus.status : "Track a request to see live updates."}
						</p>
						{jobStatus?.message && <p className="mt-1 text-xs text-muted-foreground">{jobStatus.message}</p>}
					</div>
				</div>
			</CardContent>
		</Card>
	)
}
