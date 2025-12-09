"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Box, Clock, GitBranch, Tag } from "lucide-react"
import { getMLflowRegisteredModels, type MLflowRegisteredModel } from "@/lib/api"

export default function ModelRegistry() {
  const [models, setModels] = useState<MLflowRegisteredModel[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await getMLflowRegisteredModels();
        if (res.registered_models) {
          setModels(res.registered_models);
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching registered models:", error);
        setLoading(false);
      }
    }

    fetchModels();
    const interval = setInterval(fetchModels, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card className="border-border bg-card">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm flex items-center gap-2">
            <Box className="h-4 w-4 text-primary" />
            Model Registry
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <p className="text-muted-foreground">Loading registered models...</p>
        </CardContent>
      </Card>
    );
  }

  if (models.length === 0) {
    return (
      <Card className="border-border bg-card">
        <CardHeader className="pb-4 border-b border-border">
          <CardTitle className="text-sm flex items-center gap-2">
            <Box className="h-4 w-4 text-primary" />
            Model Registry
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <p className="text-muted-foreground">No registered models found in MLflow.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-4 border-b border-border">
        <CardTitle className="text-sm flex items-center gap-2">
          <Box className="h-4 w-4 text-primary" />
          Model Registry
        </CardTitle>
        <CardDescription>Managed models in MLflow Registry</CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((model) => {
            const displayName = model.name.includes('_') ? model.name.split('_').pop() : model.name;
            return (
            <div key={model.name} className="rounded-lg border border-border bg-muted/30 p-4 hover:bg-muted/50 transition-colors">
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold text-foreground truncate" title={model.name}>
                  {displayName}
                </h3>
                <Badge variant="outline" className="text-xs">
                  v{model.latest_versions?.[0]?.version || "1"}
                </Badge>
              </div>

              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Clock className="h-3 w-3" />
                  <span>Updated: {new Date(model.last_updated_timestamp).toLocaleDateString()}</span>
                </div>

                {model.latest_versions && model.latest_versions.length > 0 && (
                  <div className="flex items-center gap-2">
                    <Tag className="h-3 w-3" />
                    <span>Stage: </span>
                    <span className={`px-1.5 py-0.5 rounded text-xs ${
                      model.latest_versions[0].current_stage === "Production"
                        ? "bg-green-500/20 text-green-400"
                        : model.latest_versions[0].current_stage === "Staging"
                          ? "bg-yellow-500/20 text-yellow-400"
                          : "bg-gray-500/20 text-gray-400"
                    }`}>
                      {model.latest_versions[0].current_stage}
                    </span>
                  </div>
                )}

                {model.description && (
                  <p className="text-xs mt-2 line-clamp-2">{model.description}</p>
                )}
              </div>
            </div>
          );
          })}
        </div>
      </CardContent>
    </Card>
  )
}
