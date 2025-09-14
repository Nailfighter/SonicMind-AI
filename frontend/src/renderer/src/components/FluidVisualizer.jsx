import React, { useRef, useEffect } from 'react'
import * as Tone from 'tone'

const FluidVisualizer = ({
  analyser1: externalAnalyser1,
  analyser2: externalAnalyser2,
  audioSource = 'file'
}) => {
  const canvasRef = useRef(null)
  const animationRef = useRef()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let analyser1
    let analyser2
    let cleanup

    const setupAudio = async () => {
      try {
        console.log('Setting up dual-channel frequency spectrum audio...')
        await Tone.start()
        console.log('Tone.js started successfully')

        // Setup channel 1
        if (externalAnalyser1 && audioSource === 'file') {
          console.log('Using external analyser for channel 1')
          analyser1 = externalAnalyser1
        } else {
          console.log('No audio source for channel 1')
          analyser1 = new Tone.Analyser('fft', 2048)
        }

        // Setup channel 2
        if (externalAnalyser2 && audioSource === 'file') {
          console.log('Using external analyser for channel 2')
          analyser2 = externalAnalyser2
        } else {
          console.log('No audio source for channel 2')
          analyser2 = new Tone.Analyser('fft', 2048)
        }

        draw()
        console.log('Dual-channel frequency spectrum audio setup completed successfully')
      } catch (error) {
        console.error('Error setting up dual-channel frequency spectrum audio:', error)
      }
    }

    const drawChannel = (analyser, channel) => {
      if (!analyser || !canvas || !ctx) return

      // Get frequency data
      const values = analyser.getValue()
      const fftSize = values.length

      // Frequency range: 20Hz to 20kHz
      const minFreq = 20
      const maxFreq = 20000
      const sampleRate = Tone.context.sampleRate
      const nyquist = sampleRate / 2

      // Convert frequency to bin index
      const freqToBin = (freq) => Math.round((freq / nyquist) * (fftSize / 2))

      const minBin = freqToBin(minFreq)
      const maxBin = Math.min(freqToBin(maxFreq), fftSize / 2)
      const usableBins = maxBin - minBin

      // Create different gradients for each channel
      let gradient, fillGradient, shadowColor

      if (channel === 1) {
        // Channel 1: Blue to cyan gradient
        gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
        gradient.addColorStop(0, 'rgba(0, 100, 255, 0.8)') // Blue
        gradient.addColorStop(0.5, 'rgba(0, 200, 255, 0.8)') // Cyan
        gradient.addColorStop(1, 'rgba(100, 255, 255, 0.8)') // Light cyan

        fillGradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
        fillGradient.addColorStop(0, 'rgba(0, 150, 255, 0.3)')
        fillGradient.addColorStop(1, 'rgba(0, 150, 255, 0.05)')

        shadowColor = 'rgba(0, 150, 255, 0.4)'
      } else {
        // Channel 2: Orange to red gradient
        gradient = ctx.createLinearGradient(0, 0, canvas.width, 0)
        gradient.addColorStop(0, 'rgba(255, 150, 0, 0.8)') // Orange
        gradient.addColorStop(0.5, 'rgba(255, 100, 50, 0.8)') // Red-orange
        gradient.addColorStop(1, 'rgba(255, 50, 100, 0.8)') // Red-pink

        fillGradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
        fillGradient.addColorStop(0, 'rgba(255, 100, 0, 0.3)')
        fillGradient.addColorStop(1, 'rgba(255, 100, 0, 0.05)')

        shadowColor = 'rgba(255, 100, 0, 0.4)'
      }

      // Create smoothed data points
      const smoothedPoints = []
      const smoothingFactor = 0.3 // Adjust for more/less smoothing

      for (let i = 0; i < usableBins; i++) {
        const binIndex = minBin + i
        let value = values[binIndex]

        if (!isFinite(value)) value = -100

        const normalizedValue = Math.max(0, (value + 100) / 100)
        const curveHeight = normalizedValue * canvas.height
        const x = (i / usableBins) * canvas.width
        const y = canvas.height - curveHeight

        // Apply smoothing by averaging with neighboring points
        let smoothedY = y
        if (i > 0 && i < usableBins - 1) {
          const prevBinIndex = minBin + i - 1
          const nextBinIndex = minBin + i + 1
          let prevValue = values[prevBinIndex]
          let nextValue = values[nextBinIndex]

          if (!isFinite(prevValue)) prevValue = -100
          if (!isFinite(nextValue)) nextValue = -100

          const prevNormalized = Math.max(0, (prevValue + 100) / 100)
          const nextNormalized = Math.max(0, (nextValue + 100) / 100)
          const prevY = canvas.height - prevNormalized * canvas.height
          const nextY = canvas.height - nextNormalized * canvas.height

          smoothedY = y * (1 - smoothingFactor) + ((prevY + nextY) / 2) * smoothingFactor
        }

        smoothedPoints.push({ x, y: smoothedY })
      }

      // Draw filled area first
      ctx.beginPath()
      ctx.moveTo(0, canvas.height)

      for (let i = 0; i < smoothedPoints.length; i++) {
        const point = smoothedPoints[i]

        if (i === 0) {
          ctx.lineTo(point.x, point.y)
        } else {
          const prevPoint = smoothedPoints[i - 1]
          const controlX = (prevPoint.x + point.x) / 2
          const controlY1 = prevPoint.y
          const controlY2 = point.y

          // Use bezier curve for smoother transitions
          ctx.bezierCurveTo(controlX, controlY1, controlX, controlY2, point.x, point.y)
        }
      }

      if (smoothedPoints.length > 0) {
        const lastPoint = smoothedPoints[smoothedPoints.length - 1]
        ctx.lineTo(canvas.width, lastPoint.y)
      }
      ctx.lineTo(canvas.width, canvas.height)
      ctx.lineTo(0, canvas.height)
      ctx.closePath()

      ctx.fillStyle = fillGradient
      ctx.fill()

      // Draw stroke outline
      ctx.strokeStyle = gradient
      ctx.lineWidth = 3
      ctx.shadowBlur = 15
      ctx.shadowColor = shadowColor

      ctx.beginPath()
      for (let i = 0; i < smoothedPoints.length; i++) {
        const point = smoothedPoints[i]

        if (i === 0) {
          ctx.moveTo(point.x, point.y)
        } else {
          const prevPoint = smoothedPoints[i - 1]
          const controlX = (prevPoint.x + point.x) / 2
          const controlY1 = prevPoint.y
          const controlY2 = point.y

          // Use bezier curve for smoother transitions
          ctx.bezierCurveTo(controlX, controlY1, controlX, controlY2, point.x, point.y)
        }
      }
      ctx.stroke()
    }

    const draw = () => {
      if (!canvas || !ctx) return

      // Clear canvas
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw both channels
      if (analyser1) {
        drawChannel(analyser1, 1)
      }
      if (analyser2) {
        drawChannel(analyser2, 2)
      }

      // Add frequency labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
      ctx.font = '14px Arial'
      ctx.shadowBlur = 0

      // 20Hz label (left side)
      ctx.textAlign = 'left'
      ctx.fillText('20Hz', 10, canvas.height - 10)

      // 20kHz label (right side)
      ctx.textAlign = 'right'
      ctx.fillText('20kHz', canvas.width - 10, canvas.height - 10)

      // Channel labels
      ctx.font = '12px Arial'
      ctx.textAlign = 'left'
      ctx.fillStyle = 'rgba(0, 150, 255, 0.8)'
      ctx.fillText('Channel 1', 10, 25)
      ctx.fillStyle = 'rgba(255, 100, 0, 0.8)'
      ctx.fillText('Channel 2', 10, 45)

      // Continue animation
      animationRef.current = requestAnimationFrame(draw)
    }

    setupAudio()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (cleanup) {
        cleanup()
      }
    }
  }, [externalAnalyser1, externalAnalyser2, audioSource])

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      style={{
        width: '100%',
        height: '100%',
        background: 'black'
      }}
    />
  )
}

export default FluidVisualizer
