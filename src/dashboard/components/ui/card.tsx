"use client"

import React from "react"

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, className, ...rest }) => (
  <div className={"rounded-lg shadow-sm " + (className || "")} {...rest}>
    {children}
  </div>
)

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, className, ...rest }) => (
  <div className={"px-4 py-3 border-b border-border " + (className || "")} {...rest}>
    {children}
  </div>
)

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, className, ...rest }) => (
  <div className={"p-4 " + (className || "")} {...rest}>
    {children}
  </div>
)

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({ children, className, ...rest }) => (
  <h3 className={"text-sm font-semibold " + (className || "")} {...rest}>
    {children}
  </h3>
)

export const CardDescription: React.FC<React.HTMLAttributes<HTMLParagraphElement>> = ({ children, className, ...rest }) => (
  <p className={"text-sm text-muted-foreground " + (className || "")} {...rest}>
    {children}
  </p>
)

export const CardFooter: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, className, ...rest }) => (
  <div className={"flex items-center p-6 pt-0 " + (className || "")} {...rest}>
    {children}
  </div>
)

export default Card
