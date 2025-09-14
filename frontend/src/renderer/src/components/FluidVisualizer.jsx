import React, { useRef, useEffect } from 'react'
import * as Tone from 'tone'

const FluidVisualizer = ({ analyser, audioContext, isPlaying }) => {
  const canvasRef = useRef(null)
  const animationRef = useRef()
  const smoothingBuffer = useRef(new Map()) // Smoothing buffer for temporal smoothing

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size to match container
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width
      canvas.height = rect.height
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)

    let analyser1
    let analyser2
    let cleanup
    let micInput

    const setupAudio = async () => {
      try {
        await Tone.start()

        // Use external analyser if provided (from video), otherwise try microphone
        if (analyser && audioContext) {
          // Use external analyser from video
          analyser1 = analyser
          analyser2 = analyser // Use same analyser for both channels
        } else {
          // Try to get microphone input
          try {
            micInput = new Tone.UserMedia()
            await micInput.open()

            // Create analysers for stereo visualization
            analyser1 = new Tone.Analyser('fft', 2048)
            analyser2 = new Tone.Analyser('fft', 2048)

            // Connect microphone to both analysers
            micInput.connect(analyser1)
            micInput.connect(analyser2)
          } catch (micError) {
            // Create analysers for procedural visualization
            analyser1 = new Tone.Analyser('fft', 2048)
            analyser2 = new Tone.Analyser('fft', 2048)
          }
        }

        draw()
      } catch (error) {
        // Silently handle setup errors
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
      const numPoints = Math.min(canvas.width / 1.5, 400) // Higher resolution

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

      // Create smoothed data points using logarithmic frequency mapping
      const smoothedPoints = []
      const smoothingFactor = 0.7 // Temporal smoothing
      const spatialSmoothing = 3 // Spatial smoothing window

      // Get or create smoothing buffer for this channel
      const channelKey = `channel_${channel}`
      if (!smoothingBuffer.current.has(channelKey)) {
        smoothingBuffer.current.set(channelKey, new Array(numPoints).fill(-100))
      }
      const prevValues = smoothingBuffer.current.get(channelKey)

      for (let i = 0; i < numPoints; i++) {
        // Calculate frequency mapping optimized for musical content
        const normalizedPos = i / (numPoints - 1) // 0 to 1
        let freq

        if (normalizedPos <= 0.15) {
          // First 15% covers 20Hz to 200Hz (bass/sub-bass)
          const subPos = normalizedPos / 0.15
          freq = 20 + 180 * Math.pow(subPos, 0.7) // Slightly compressed bass
        } else if (normalizedPos <= 0.75) {
          // Next 60% covers 200Hz to 4kHz (critical midrange for vocals/instruments)
          const subPos = (normalizedPos - 0.15) / 0.6
          freq = 200 + 3800 * subPos // Linear high-resolution mapping for mids
        } else {
          // Last 25% covers 4kHz to 20kHz (presence/air)
          const subPos = (normalizedPos - 0.75) / 0.25
          freq = 4000 * Math.pow(5, subPos) // Logarithmic for highs
        }

        // Map frequency to FFT bin
        const binIndex = Math.floor((freq / nyquist) * (fftSize / 2))
        const clampedIndex = Math.max(minBin, Math.min(maxBin - 1, binIndex))

        let value = values[clampedIndex]
        if (!isFinite(value)) value = -100

        // Apply spatial smoothing with neighboring bins
        let smoothedValue = 0
        let count = 0
        for (let j = -spatialSmoothing; j <= spatialSmoothing; j++) {
          const idx = Math.max(minBin, Math.min(maxBin - 1, clampedIndex + j))
          let neighborValue = values[idx]
          if (!isFinite(neighborValue)) neighborValue = -100
          smoothedValue += neighborValue
          count++
        }
        smoothedValue /= count

        // Apply temporal smoothing
        const temporalSmoothed =
          prevValues[i] * smoothingFactor + smoothedValue * (1 - smoothingFactor)
        prevValues[i] = temporalSmoothed

        // Convert dB to normalized value (-100dB to 0dB -> 0 to 1)
        const normalizedValue = Math.max(0, (temporalSmoothed + 100) / 100)
        const curveHeight = normalizedValue * (canvas.height - 50) * 0.9
        const x = (i / (numPoints - 1)) * canvas.width
        const y = canvas.height - 50 - curveHeight

        smoothedPoints.push({ x, y, db: temporalSmoothed })
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

      // Calculate peak level for display
      const peakDb = Math.max(...smoothedPoints.map((p) => p.db))

      // Draw peak indicators on the curve
      ctx.shadowBlur = 0
      ctx.fillStyle = gradient
      smoothedPoints.forEach((point, i) => {
        if (point.db > -6 && i % 15 === 0) {
          // Show peaks above -6dB
          ctx.beginPath()
          ctx.arc(point.x, point.y, 2, 0, Math.PI * 2)
          ctx.fill()
        }
      })

      // Draw channel label with peak level
      ctx.font = '11px Arial'
      ctx.textAlign = 'left'
      const peakColor =
        peakDb > -6
          ? '#ff6666'
          : peakDb > -12
            ? '#ffaa00'
            : channel === 1
              ? 'rgba(0, 150, 255, 0.8)'
              : 'rgba(255, 100, 0, 0.8)'
      ctx.fillStyle = peakColor
      ctx.fillText(`Channel ${channel} (Peak: ${peakDb.toFixed(1)}dB)`, 10, 20 + (channel - 1) * 15)
    }

    const drawProcedural = () => {
      if (!canvas || !ctx) return

      const time = Date.now() * 0.001
      const points1 = []
      const points2 = []

      // Generate procedural waveforms
      for (let i = 0; i < 100; i++) {
        const x = (i / 99) * canvas.width
        const freq1 = 0.02 + i * 0.001
        const freq2 = 0.015 + i * 0.0008

        const y1 = canvas.height * 0.7 + Math.sin(time * freq1 + i * 0.1) * canvas.height * 0.2
        const y2 = canvas.height * 0.3 + Math.cos(time * freq2 + i * 0.15) * canvas.height * 0.15

        points1.push({ x, y: y1 })
        points2.push({ x, y: y2 })
      }

      // Draw channel 1 (blue)
      ctx.strokeStyle = 'rgba(0, 150, 255, 0.8)'
      ctx.lineWidth = 3
      ctx.shadowBlur = 15
      ctx.shadowColor = 'rgba(0, 150, 255, 0.4)'

      ctx.beginPath()
      points1.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y)
        else ctx.lineTo(point.x, point.y)
      })
      ctx.stroke()

      // Draw channel 2 (orange)
      ctx.strokeStyle = 'rgba(255, 100, 0, 0.8)'
      ctx.shadowColor = 'rgba(255, 100, 0, 0.4)'

      ctx.beginPath()
      points2.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y)
        else ctx.lineTo(point.x, point.y)
      })
      ctx.stroke()
    }

    const draw = () => {
      if (!canvas || !ctx) return

      // Clear canvas with dark grey background
      ctx.fillStyle = '#2a2a2a'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Main frequency labels
      ctx.fillStyle = '#ffffff'
      ctx.font = '12px Arial'
      ctx.textAlign = 'left'
      ctx.fillText('20Hz', 10, canvas.height - 10)
      ctx.textAlign = 'right'
      ctx.fillText('20kHz', canvas.width - 10, canvas.height - 10)

      // Draw detailed frequency grid
      ctx.strokeStyle = '#333333'
      ctx.lineWidth = 1
      ctx.setLineDash([1, 3])

      // Major frequency markers optimized for musical content
      const majorFreqs = [50, 100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000, 8000, 16000]
      majorFreqs.forEach((freq) => {
        // Calculate x position using the same musical frequency mapping
        let x
        if (freq <= 200) {
          const subPos = Math.pow((freq - 20) / 180, 1 / 0.7)
          x = subPos * 0.15 * canvas.width
        } else if (freq <= 4000) {
          const subPos = (freq - 200) / 3800
          x = (0.15 + subPos * 0.6) * canvas.width
        } else {
          const subPos = Math.log(freq / 4000) / Math.log(5)
          x = (0.75 + subPos * 0.25) * canvas.width
        }
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, canvas.height - 50)
        ctx.stroke()

        // Major frequency labels
        ctx.fillStyle = '#cccccc'
        ctx.font = '10px Arial'
        ctx.textAlign = 'center'
        let label
        if (freq >= 1000) {
          label = freq / 1000 + 'k'
        } else {
          label = freq + ''
        }
        ctx.fillText(label, x, canvas.height - 35)
      })

      // Minor frequency markers
      ctx.strokeStyle = '#222222'
      ctx.setLineDash([1, 5])
      const minorFreqs = [30, 60, 150, 300, 600, 1000, 1400, 1800, 2200, 2800, 3500, 6000, 12000]
      minorFreqs.forEach((freq) => {
        // Calculate x position using the same musical frequency mapping
        let x
        if (freq <= 200) {
          const subPos = Math.pow((freq - 20) / 180, 1 / 0.7)
          x = subPos * 0.15 * canvas.width
        } else if (freq <= 4000) {
          const subPos = (freq - 200) / 3800
          x = (0.15 + subPos * 0.6) * canvas.width
        } else {
          const subPos = Math.log(freq / 4000) / Math.log(5)
          x = (0.75 + subPos * 0.25) * canvas.width
        }
        ctx.beginPath()
        ctx.moveTo(x, canvas.height - 50)
        ctx.lineTo(x, canvas.height - 45)
        ctx.stroke()
      })

      // Draw decibel scale
      ctx.setLineDash([])
      ctx.strokeStyle = '#444444'
      ctx.lineWidth = 1

      // Horizontal dB grid lines
      const dbLevels = [-60, -40, -20, -10, -6, -3, 0]
      dbLevels.forEach((db) => {
        const y = canvas.height - 50 - ((db + 60) / 60) * (canvas.height - 50)
        if (y >= 0 && y <= canvas.height - 50) {
          ctx.beginPath()
          ctx.moveTo(0, y)
          ctx.lineTo(canvas.width, y)
          ctx.stroke()

          // dB labels
          ctx.fillStyle = db === 0 ? '#ff6666' : '#999999'
          ctx.font = '9px Arial'
          ctx.textAlign = 'right'
          ctx.fillText(db + 'dB', canvas.width - 5, y - 2)
        }
      })

      ctx.setLineDash([])

      // Draw based on available input
      if (analyser && audioContext && analyser1) {
        // Draw from video audio
        drawChannel(analyser1, 1)
        drawChannel(analyser1, 2) // Use same analyser for both channels
      } else if (micInput && analyser1 && analyser2) {
        // Draw both channels from microphone
        drawChannel(analyser1, 1)
        drawChannel(analyser2, 2)
      } else {
        // Draw procedural visualization
        drawProcedural()
      }

      // Add frequency labels
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'
      ctx.font = '14px Arial'
      ctx.shadowBlur = 0

      // Status label
      ctx.textAlign = 'left'
      if (analyser && audioContext) {
        ctx.fillText('Video Audio Input', 10, canvas.height - 10)
      } else if (micInput) {
        ctx.fillText('Microphone Input', 10, canvas.height - 10)
      } else {
        ctx.fillText('Procedural Visualization', 10, canvas.height - 10)
      }

      // Channel labels are now drawn with peak levels in drawChannel function

      // Continue animation
      animationRef.current = requestAnimationFrame(draw)
    }

    setupAudio()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      window.removeEventListener('resize', resizeCanvas)
      if (cleanup) {
        cleanup()
      }
      if (micInput) {
        micInput.close()
      }
    }
  }, [analyser, audioContext, isPlaying])

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: '100%',
        height: '100%',
        background: 'black',
        display: 'block'
      }}
    />
  )
}

export default FluidVisualizer
