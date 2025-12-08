"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, AlertCircle, CheckCircle2 } from "lucide-react"
import { submitDataset } from "../../lib/api"
import AppLayout from "../../components/app-layout"

export default function UploadPage() {
  const router = useRouter()
  const [datasetName, setDatasetName] = useState("")
  const [experimentName, setExperimentName] = useState("")
  const [targetName, setTargetName] = useState("")
  const [taskType, setTaskType] = useState<"classification" | "regression">("classification")
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.type !== "text/csv" && !selectedFile.name.endsWith(".csv")) {
        setError("Please select a CSV file")
        setFile(null)
        return
      }
      setFile(selectedFile)
      setError(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setSuccess(null)

    if (!file) {
      setError("Please select a CSV file")
      return
    }

    if (!datasetName.trim()) {
      setError("Please enter a dataset name")
      return
    }

    if (!targetName.trim()) {
      setError("Please enter a target column name")
      return
    }

    setUploading(true)

    try {
      // Read file as text
      const text = await file.text()

      // Submit to API
      const response = await submitDataset({
        name: datasetName.trim(),
        target_name: targetName.trim(),
        task_type: taskType,
        dataset_csv: text,
        mlflow_experiment: experimentName.trim() || undefined,
      })

      setSuccess(`Job started successfully! Request ID: ${response.request_id}`)

      // Redirect to dashboard with the request ID after 2 seconds
      setTimeout(() => {
        router.push(`/?requestId=${encodeURIComponent(response.request_id)}`)
      }, 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload dataset")
    } finally {
      setUploading(false)
    }
  }

  return (
    <AppLayout>
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-foreground">Upload Dataset</h1>
          <Button variant="ghost" onClick={() => router.push("/")}>
            Back to Dashboard
          </Button>
        </div>

        <Card className="border-border bg-card">
          <CardHeader>
            <CardTitle>New Training Job</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="p-3 text-sm text-red-500 bg-red-500/10 border border-red-500/20 rounded-md flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </div>
              )}

              {success && (
                <div className="p-3 text-sm text-green-500 bg-green-500/10 border border-green-500/20 rounded-md flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  {success}
                </div>
              )}

              <div className="space-y-2">
                <label className="text-sm font-medium">Dataset Name</label>
                <input
                  type="text"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  placeholder="e.g., Iris Classification"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Experiment Name (Optional)</label>
                <input
                  type="text"
                  value={experimentName}
                  onChange={(e) => setExperimentName(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  placeholder="e.g., My Experiment"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Target Column</label>
                <input
                  type="text"
                  value={targetName}
                  onChange={(e) => setTargetName(e.target.value)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  placeholder="e.g., species"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Task Type</label>
                <select
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value as "classification" | "regression")}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Dataset File (CSV)</label>
                <div className="flex items-center justify-center w-full">
                  <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-lg cursor-pointer hover:bg-accent/50 border-border">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload className="w-8 h-8 mb-4 text-muted-foreground" />
                      <p className="mb-2 text-sm text-muted-foreground">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-muted-foreground">CSV file only</p>
                    </div>
                    <input
                      type="file"
                      className="hidden"
                      accept=".csv"
                      onChange={handleFileChange}
                    />
                  </label>
                </div>
                {file && (
                  <p className="text-sm text-muted-foreground">
                    Selected: {file.name}
                  </p>
                )}
              </div>

              <Button type="submit" className="w-full" disabled={uploading}>
                {uploading ? "Uploading..." : "Start Training Job"}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  )
}
