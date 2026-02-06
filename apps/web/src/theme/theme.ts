import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'dark',
    background: {
      default: '#131722',
      paper: '#1e222d',
    },
    primary: {
      main: '#2962ff',
    },
    secondary: {
      main: '#b2b5be',
    },
    success: {
      main: '#089981',
    },
    error: {
      main: '#f23645',
    },
    warning: {
      main: '#f0b90b',
    },
    text: {
      primary: '#d1d4dc',
      secondary: '#787b86',
    },
    divider: '#2a2e39',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 600,
    },
    subtitle1: {
      fontWeight: 500,
    },
    body1: {
      fontSize: '0.925rem',
    },
    body2: {
      fontSize: '0.85rem',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#131722',
          scrollbarColor: '#2a2e39 #131722',
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: '#131722',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#2a2e39',
            borderRadius: '4px',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
  },
})
