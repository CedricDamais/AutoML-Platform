const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const MLFLOW_BASE_URL = process.env.NEXT_PUBLIC_MLFLOW_URL || "http://localhost:5001";

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

async function request<T>(path: string, options: { method?: HttpMethod; body?: unknown } = {}): Promise<T> {
  const { method = "GET", body } = options;
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });

  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    const error = typeof detail === "object" && detail !== null ? JSON.stringify(detail) : res.statusText;
    throw new Error(`Request failed: ${res.status} ${error}`);
  }

  return res.json() as Promise<T>;
}

export interface DatasetRequestPayload {
  name: string;
  target_name: string;
  task_type?: "classification" | "regression";
  dataset_csv: string;
  mlflow_experiment?: string;
}

export interface DatasetResponse {
  message: string;
  request_id: string;
  status_url: string;
}

export async function submitDataset(payload: DatasetRequestPayload) {
  return request<DatasetResponse>("/api/v1/d_dataset", { method: "POST", body: payload });
}

export interface JobStatusResponse {
  status: string;
  dataset_name?: string;
  created_at?: string;
  message?: string;
  [key: string]: unknown;
}

export async function getJobStatus(requestId: string) {
  return request<JobStatusResponse>(`/api/v1/jobs/${requestId}`);
}

export interface JobSummary {
  request_id: string;
  dataset_name: string;
  experiment_name?: string;
  status: string;
  created_at: string;
}

export interface JobsListResponse {
  jobs: JobSummary[];
}

export async function getJobs() {
  return request<JobsListResponse>("/api/v1/jobs");
}

export interface HealthResponse {
  status: string;
}

export async function getHealth() {
  return request<HealthResponse>("/health");
}

// MLflow API Types
export interface MLflowMetric {
  key: string;
  value: number;
  timestamp?: number;
  step?: number;
}

export interface MLflowParam {
  key: string;
  value: string;
}

export interface MLflowTag {
  key: string;
  value: string;
}

export interface MLflowRunData {
  metrics: MLflowMetric[];
  params: MLflowParam[];
  tags: MLflowTag[];
}

export interface MLflowRunInfo {
  run_id: string;
  run_name: string;
  experiment_id: string;
  status: string;
  start_time?: number;
  end_time?: number;
}

export interface MLflowRun {
  info: MLflowRunInfo;
  data: MLflowRunData;
}

export interface MLflowRunsResponse {
  runs: MLflowRun[];
}

export interface MLflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location?: string;
  lifecycle_stage?: string;
}

export interface MLflowExperimentsResponse {
  experiments: MLflowExperiment[];
}

export async function getMLflowExperiments() {
  const res = await fetch(`${MLFLOW_BASE_URL}/ajax-api/2.0/mlflow/experiments/search?max_results=1000`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch MLflow experiments");
  return res.json() as Promise<MLflowExperimentsResponse>;
}

export async function getMLflowRuns(experimentId: string) {
  const res = await fetch(`${MLFLOW_BASE_URL}/api/2.0/mlflow/runs/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      experiment_ids: [experimentId],
      max_results: 100,
      order_by: ["start_time DESC"],
    }),
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch MLflow runs");
  return res.json() as Promise<MLflowRunsResponse>;
}

export interface MLflowRegisteredModel {
  name: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  description?: string;
  latest_versions?: {
    version: string;
    creation_timestamp: number;
    current_stage: string;
    source: string;
    run_id: string;
    status: string;
  }[];
}

export interface MLflowRegisteredModelsResponse {
  registered_models: MLflowRegisteredModel[];
}

export async function getMLflowRegisteredModels() {
  const res = await fetch(`${MLFLOW_BASE_URL}/api/2.0/mlflow/registered-models/search`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch MLflow registered models");
  return res.json() as Promise<MLflowRegisteredModelsResponse>;
}

export interface MLflowMetricHistoryResponse {
  metrics: MLflowMetric[];
}

export async function getMLflowMetricHistory(runId: string, metricKey: string) {
  const res = await fetch(`${MLFLOW_BASE_URL}/ajax-api/2.0/mlflow/metrics/get-history?run_id=${runId}&metric_key=${metricKey}`, {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to fetch MLflow metric history");
  return res.json() as Promise<MLflowMetricHistoryResponse>;
}
