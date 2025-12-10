import { CheckCircle2, Circle, Loader2 } from "lucide-react";

interface PipelineProps {
    currentStep: string;
    steps: { id: string; label: string }[];
}

export function DeploymentPipeline({ currentStep, steps }: PipelineProps) {
    // Determine the index of the current step
    // If currentStep is "DONE", all are completed.
    // If currentStep is "FAILED", we might want to show that, but for now let's focus on progress.

    const isDone = currentStep === "DONE" || currentStep === "SUCCESS";

    const getCurrentStepIndex = () => {
        if (isDone) return steps.length;
        return steps.findIndex(s => s.id === currentStep);
    };

    const currentIndex = getCurrentStepIndex();

    return (
        <div className="w-full py-4">
            <div className="relative flex items-center justify-between w-full">
                {/* Connecting Line Background */}
                <div className="absolute left-0 top-1/2 transform -translate-y-1/2 w-full h-1 bg-muted -z-10" />

                {/* Connecting Line Progress */}
                <div
                    className="absolute left-0 top-1/2 transform -translate-y-1/2 h-1 bg-primary -z-10 transition-all duration-500 ease-in-out"
                    style={{ width: `${(Math.max(0, currentIndex) / (steps.length - 1)) * 100}%` }}
                />

                {steps.map((step, index) => {
                    let status: "completed" | "current" | "pending" = "pending";

                    if (isDone || index < currentIndex) {
                        status = "completed";
                    } else if (index === currentIndex) {
                        status = "current";
                    }

                    return (
                        <div key={step.id} className="flex flex-col items-center gap-2 bg-card z-10 px-2">
                            <div className={`
                  flex items-center justify-center w-8 h-8 rounded-full border-2 transition-all duration-300
                  ${status === "completed" ? "bg-primary border-primary text-primary-foreground" : ""}
                  ${status === "current" ? "border-primary text-primary bg-background animate-pulse" : ""}
                  ${status === "pending" ? "border-muted text-muted-foreground bg-background" : ""}
               `}>
                                {status === "completed" && <CheckCircle2 className="w-5 h-5" />}
                                {status === "current" && <Loader2 className="w-4 h-4 animate-spin" />}
                                {status === "pending" && <span className="text-xs font-medium">{index + 1}</span>}
                            </div>
                            <span className={`text-xs font-medium ${status === "pending" ? "text-muted-foreground" : "text-foreground"}`}>
                                {step.label}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
