import { Box } from '@mui/material'

import { TVChartContainer } from './TVChartContainer'
import { TVTickerTape } from './TVTickerTape'

interface TVDashboardProps {
    symbol: string
    interval?: string
    theme?: 'light' | 'dark'
}

export const TVDashboard: React.FC<TVDashboardProps> = ({
    symbol,
    interval = 'D',
    theme = 'dark',
}) => {
    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                width: '100%',
                height: '100%',
            }}
        >
            <Box sx={{ height: 46, width: '100%', overflow: 'hidden' }}>
                <TVTickerTape theme={theme} />
            </Box>

            <Box sx={{ flexGrow: 1, minHeight: 600, position: 'relative' }}>
                <TVChartContainer symbol={symbol} theme={theme} interval={interval} />
            </Box>
        </Box>
    )
}
