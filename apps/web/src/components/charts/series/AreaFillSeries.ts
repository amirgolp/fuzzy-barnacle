import {
  type CustomSeriesOptions, // Import CustomSeriesOptions
  type CustomSeriesPricePlotValues,
  type ICustomSeriesPaneView,
  type PaneRendererCustomData,
  type Time,
  type WhitespaceData,
} from 'lightweight-charts'

export interface AreaFillData {
  time: Time
  value1: number
  value2: number
  color: string
}

// Extend CustomSeriesOptions to ensure compatibility
export interface AreaFillSeriesOptions extends CustomSeriesOptions {
  lineColor?: string
  lineWidth?: number
  // Add missing properties expected by the linter if CustomSeriesOptions doesn't cover them all transparently
  color: string // Required by CustomStyleOptions
}

class AreaFillPaneRenderer {
  _data: PaneRendererCustomData<Time, AreaFillData> | null = null
  _options: AreaFillSeriesOptions | null = null

  draw(target: any, priceConverter: any) {
    this._drawImpl(target, priceConverter)
  }

  _drawImpl(target: any, priceConverter: any) {
    const data = this._data
    if (!data || data.bars.length === 0 || !data.visibleRange) return

    target.useBitmapCoordinateSpace((scope: any) => {
      const ctx = scope.context

      const bars = data.bars

      // Group bars by color
      let currentPath: { x: number; y1: number; y2: number }[] = []
      let currentColor: string | null = null

      for (let i = 0; i < bars.length; i++) {
        const bar = bars[i]
        const x = bar.x
        const item = bar.originalData as AreaFillData

        const y1 = priceConverter.priceToCoordinate(item.value1)
        const y2 = priceConverter.priceToCoordinate(item.value2)

        if (y1 === null || y2 === null) continue

        if (item.color !== currentColor) {
          if (currentPath.length > 0 && currentColor) {
            this._drawPath(ctx, currentPath, currentColor)
          }
          currentPath = []
          currentColor = item.color
        }

        currentPath.push({ x, y1, y2 })
      }

      if (currentPath.length > 0 && currentColor) {
        this._drawPath(ctx, currentPath, currentColor)
      }
    })
  }

  _drawPath(
    ctx: CanvasRenderingContext2D,
    points: { x: number; y1: number; y2: number }[],
    color: string,
  ) {
    if (points.length === 0) return

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y1)

    // Top line
    for (const p of points) {
      ctx.lineTo(p.x, p.y1)
    }

    // Bottom line
    for (let i = points.length - 1; i >= 0; i--) {
      ctx.lineTo(points[i].x, points[i].y2)
    }

    ctx.closePath()
    ctx.fill()
  }
}

export class AreaFillSeries implements ICustomSeriesPaneView<
  Time,
  AreaFillData,
  AreaFillSeriesOptions
> {
  _renderer: AreaFillPaneRenderer

  constructor() {
    this._renderer = new AreaFillPaneRenderer()
  }

  priceValueBuilder(plotRow: AreaFillData): CustomSeriesPricePlotValues {
    return [plotRow.value1, plotRow.value2]
  }

  isWhitespace(data: AreaFillData | WhitespaceData): data is WhitespaceData {
    return (data as Partial<AreaFillData>).value1 === undefined
  }

  renderer(): AreaFillPaneRenderer {
    return this._renderer
  }

  update(
    data: PaneRendererCustomData<Time, AreaFillData>,
    options: AreaFillSeriesOptions,
  ): void {
    this._renderer._data = data
    this._renderer._options = options
  }

  defaultOptions(): AreaFillSeriesOptions {
    return {
      color: 'transparent',
      lineColor: '#26a69a',
      lineWidth: 2,
      visible: true,
      lastValueVisible: false,
      priceLineVisible: false,
    } as AreaFillSeriesOptions
  }
}
