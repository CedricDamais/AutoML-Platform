"use client"

import AppLayout from "../../components/app-layout"
import ModelRegistry from "../../components/model_registry"

export default function RegistryPage() {
  return (
    <AppLayout>
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-foreground">Model Registry</h1>
        <ModelRegistry />
      </div>
    </AppLayout>
  )
}
