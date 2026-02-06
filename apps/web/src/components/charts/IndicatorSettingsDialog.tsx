import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  Switch,
  TextField,
} from '@mui/material'
import React, { useEffect, useState } from 'react'

interface IndicatorSettingsDialogProps {
  open: boolean
  indicatorName: string
  params: Record<string, any>
  onClose: () => void
  onSave: (name: string, newParams: Record<string, any>) => void
}

export const IndicatorSettingsDialog: React.FC<
  IndicatorSettingsDialogProps
> = ({ open, indicatorName, params, onClose, onSave }) => {
  const [localParams, setLocalParams] = useState<Record<string, any>>({})

  useEffect(() => {
    if (open) {
      setLocalParams({ ...params })
    }
  }, [open, params])

  const handleChange = (key: string, value: any) => {
    setLocalParams((prev) => ({ ...prev, [key]: value }))
  }

  const handleSave = () => {
    onSave(indicatorName, localParams)
    onClose()
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      PaperProps={{
        style: {
          backgroundColor: '#1e222d',
          color: '#d1d4dc',
          minWidth: '300px',
        },
      }}
    >
      <DialogTitle sx={{ borderBottom: '1px solid #2a2e39' }}>
        {indicatorName} Settings
      </DialogTitle>
      <DialogContent sx={{ mt: 2 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {Object.entries(localParams).map(([key, value]) => {
            if (typeof value === 'boolean') {
              return (
                <FormControlLabel
                  key={key}
                  control={
                    <Switch
                      checked={value}
                      onChange={(e) => handleChange(key, e.target.checked)}
                      color="primary"
                    />
                  }
                  label={key}
                  sx={{ color: '#d1d4dc' }}
                />
              )
            } else if (typeof value === 'number') {
              return (
                <TextField
                  key={key}
                  label={key}
                  type="number"
                  value={value}
                  onChange={(e) =>
                    handleChange(key, parseFloat(e.target.value))
                  }
                  variant="outlined"
                  size="small"
                  margin="dense"
                  InputLabelProps={{ shrink: true, sx: { color: '#888' } }}
                  InputProps={{
                    sx: {
                      color: '#d1d4dc',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#444',
                      },
                    },
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#666',
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#2962FF',
                      },
                    },
                  }}
                />
              )
            } else {
              return (
                <TextField
                  key={key}
                  label={key}
                  type="text"
                  value={value}
                  onChange={(e) => handleChange(key, e.target.value)}
                  variant="outlined"
                  size="small"
                  margin="dense"
                  InputLabelProps={{ shrink: true, sx: { color: '#888' } }}
                  InputProps={{
                    sx: {
                      color: '#d1d4dc',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#444',
                      },
                    },
                  }}
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      '&:hover .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#666',
                      },
                      '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                        borderColor: '#2962FF',
                      },
                    },
                  }}
                />
              )
            }
          })}
        </Box>
      </DialogContent>
      <DialogActions sx={{ borderTop: '1px solid #2a2e39', p: 2 }}>
        <Button onClick={onClose} sx={{ color: '#888' }}>
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          sx={{ bgcolor: '#2962FF' }}
        >
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  )
}
