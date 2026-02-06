import { useEffect, useRef } from 'react'

interface TVTickerTapeProps {
    theme?: 'light' | 'dark'
}

export const TVTickerTape = ({ theme = 'dark' }: TVTickerTapeProps) => {
    const containerRef = useRef<HTMLDivElement>(null)

    useEffect(() => {
        if (!containerRef.current) return

        // Clear previous content to avoid duplicates on re-renders (though React handles this mostly)
        containerRef.current.innerHTML = ''

        const script = document.createElement('script')
        script.src =
            'https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js'
        script.async = true
        script.innerHTML = JSON.stringify({
            symbols: [
                { proName: 'FOREXCOM:SPXUSD', title: 'S&P 500' },
                { proName: 'FOREXCOM:NSXUSD', title: 'US 100' },
                { proName: 'FX_IDC:EURUSD', title: 'EUR/USD' },
                { proName: 'BITSTAMP:BTCUSD', title: 'BTC/USD' },
                { proName: 'BITSTAMP:ETHUSD', title: 'ETH/USD' },
            ],
            showSymbolLogo: true,
            isTransparent: true,
            displayMode: 'adaptive',
            colorTheme: theme,
            locale: 'en',
        })

        containerRef.current.appendChild(script)
    }, [theme])

    return (
        <div className="tradingview-widget-container" ref={containerRef}>
            <div className="tradingview-widget-container__widget"></div>
        </div>
    )
}
