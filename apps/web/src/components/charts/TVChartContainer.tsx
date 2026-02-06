import { useEffect, useRef } from 'react'

interface TVChartContainerProps {
    symbol: string
    theme?: 'light' | 'dark'
    interval?: string
}

declare global {
    interface Window {
        TradingView: any
    }
}

export const TVChartContainer = ({
    symbol,
    theme = 'dark',
    interval = 'D',
}: TVChartContainerProps) => {
    const containerRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        let tvScript: HTMLScriptElement | null = null

        const initWidget = () => {
            if (containerRef.current && window.TradingView) {
                new window.TradingView.widget({
                    autosize: true,
                    symbol: symbol,
                    interval: interval,
                    timezone: 'Etc/UTC',
                    theme: theme,
                    style: '1',
                    locale: 'en',
                    toolbar_bg: '#f1f3f6',
                    enable_publishing: false,
                    allow_symbol_change: true,
                    container_id: containerRef.current.id,
                    studies: [
                        'MACD@tv-basicstudies',
                        'StochasticRSI@tv-basicstudies',
                        'TripleEMA@tv-basicstudies',
                    ],
                })
            }
        }

        if (!document.getElementById('tradingview-widget-loading-script')) {
            tvScript = document.createElement('script')
            tvScript.id = 'tradingview-widget-loading-script'
            tvScript.src = 'https://s3.tradingview.com/tv.js'
            tvScript.type = 'text/javascript'
            tvScript.onload = initWidget
            document.head.appendChild(tvScript)
        } else {
            // Script already exists, verify if loaded or wait
            if (window.TradingView) {
                initWidget()
            } else {
                // Find existing script and attach onload if not loaded? 
                // Simpler: just poll or wait for the existing one. 
                // But usually, it loads fast. Let's try adding a listener to the existing script if possible
                // or just set timeout.
                const existingScript = document.getElementById('tradingview-widget-loading-script') as HTMLScriptElement
                if (existingScript) {
                    const originalOnLoad = existingScript.onload;
                    existingScript.onload = (e) => {
                        if (originalOnLoad) (originalOnLoad as any)(e);
                        initWidget();
                    }
                }
            }
        }

        // Quick fallback helper
        const checkInterval = setInterval(() => {
            if (window.TradingView && containerRef.current && containerRef.current.innerHTML === "") {
                initWidget();
                clearInterval(checkInterval);
            }
        }, 500);

        return () => {
            clearInterval(checkInterval);
            // Cleanup if necessary? TradingView widget replaces content, so react re-render handles it usually.
        }
    }, [symbol, theme, interval])

    return (
        <div
            id={`tradingview_chart_${Math.random().toString(36).substring(7)}`}
            ref={containerRef}
            style={{ height: '100%', width: '100%' }}
        />
    )
}
