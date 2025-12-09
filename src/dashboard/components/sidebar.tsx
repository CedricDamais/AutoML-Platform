"use client"

import { useState } from "react"
import { useRouter, usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import {
  Menu,
  LayoutDashboard,
  Database,
  Flag as Flask,
  History as Registry,
  Zap,
  Settings,
  User,
  LogOut,
} from "lucide-react"

interface SidebarProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: Database, label: "Upload Dataset", href: "/upload" },
  { icon: Registry, label: "Model Registry", href: "/registry" },
]

export default function Sidebar({ open, onOpenChange }: SidebarProps) {
  const router = useRouter()
  const pathname = usePathname()

  return (
    <aside
      className={`bg-sidebar border-r border-sidebar-border transition-all duration-300 ${open ? "w-64" : "w-20"} flex flex-col`}
    >
      {/* Header */}
      <div className="p-4 border-b border-sidebar-border flex items-center justify-between">
        {open && <h1 className="text-lg font-semibold text-sidebar-foreground">AutoML</h1>}
        <Button variant="ghost" size="icon" onClick={() => onOpenChange(!open)} className="h-8 w-8">
          <Menu className="h-4 w-4" />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <button
              key={item.label}
              onClick={() => item.href !== "#" && router.push(item.href)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
                isActive
                  ? "bg-sidebar-primary/20 text-sidebar-primary"
                  : "text-sidebar-foreground hover:bg-sidebar-accent"
              }`}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {open && <span className="text-sm font-medium">{item.label}</span>}
            </button>
          )
        })}
      </nav>

      {/* User Profile */}
      <div className="border-t border-sidebar-border p-4 space-y-2">
        <div className="flex items-center gap-3 px-3 py-2">
          <div className="h-8 w-8 rounded-full bg-sidebar-primary/20 flex items-center justify-center flex-shrink-0">
            <User className="h-4 w-4 text-sidebar-primary" />
          </div>
          {open && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-sidebar-foreground truncate">Dr. Sarah Chen</p>
              <p className="text-xs text-sidebar-primary">Data Scientist</p>
            </div>
          )}
        </div>
        {open && (
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent"
          >
            <LogOut className="h-4 w-4 mr-2" />
            Logout
          </Button>
        )}
      </div>
    </aside>
  )
}
