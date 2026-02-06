import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo)
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '32px', backgroundColor: '#7f1d1d', color: '#fff' }}>
          <h1
            style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              marginBottom: '16px',
            }}
          >
            Something went wrong
          </h1>
          <pre
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.3)',
              padding: '16px',
              borderRadius: '4px',
              overflow: 'auto',
            }}
          >
            {this.state.error?.toString()}
            {this.state.error?.stack}
          </pre>
        </div>
      )
    }

    return this.props.children
  }
}
