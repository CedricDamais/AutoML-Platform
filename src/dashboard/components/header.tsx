"use client"

import { Button } from "@/components/ui/button"
import { Plus, FileText, Bell, Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

interface HeaderProps {
  onMenuClick: () => void
}

export default function Header({ onMenuClick }: HeaderProps) {
  const { theme, setTheme } = useTheme()
  return (
    <header className="border-b border-border bg-card h-16 flex items-center px-6 gap-4">
      {/* Breadcrumbs */}
      <div className="flex-1 flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Projects</span>
        <span className="text-muted-foreground">/</span>
        <span className="text-foreground font-medium">Credit Risk Model</span>
        <span className="text-muted-foreground">/</span>
        <span className="text-primary font-medium">Experiment #42</span>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" className="gap-2 bg-transparent">
          <Plus className="h-4 w-4" />
          New Experiment
        </Button>
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <FileText className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" className="h-9 w-9 relative">
          <Bell className="h-4 w-4" />
          <span className="absolute top-1 right-1 h-2 w-2 bg-red-500 rounded-full" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-9 w-9"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </div>
    </header>
  )
}
