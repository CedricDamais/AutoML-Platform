"use client"

import React from "react"
import clsx from "clsx"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "ghost" | "outline" | "default"
  size?: "icon" | "sm" | "md"
}

export const Button: React.FC<ButtonProps> = ({
  variant = "default",
  size = "md",
  className,
  children,
  ...rest
}) => {
  const base = "inline-flex items-center justify-center rounded-md font-medium focus:outline-none"
  const variantClass = variant === "ghost" ? "bg-transparent hover:bg-muted" : variant === "outline" ? "border border-border" : "bg-primary text-primary-foreground"
  const sizeClass = size === "icon" ? "p-2 h-9 w-9" : size === "sm" ? "px-2 py-1 text-sm" : "px-4 py-2"

  return (
    <button className={clsx(base, variantClass, sizeClass, className)} {...rest}>
      {children}
    </button>
  )
}

export default Button
