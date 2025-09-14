import React, { useState, useRef, useEffect } from 'react'
import Knob from './Knob'
import Slider from './Slider'
import FluidVisualizer from './FluidVisualizer'
import apiService from '../services/apiService'

const LiveSettingPage = ({ profile, onBack }) => {
  // State for effect knob values
  const [effectValues, setEffectValues] = useState({
    reverb: 0,
    delay: 0,
    overdrive: 0,
    fuzz: 0,
    compression: 0,
    gain: 0
  })

  // State for EQ slider values (5 bands matching backend)
  const [eqValues, setEqValues] = useState([
    { freq: 80, gain: 0, q: 1.0 },     // Low
    { freq: 300, gain: 0, q: 1.2 },   // Low-Mid
    { freq: 1000, gain: 0, q: 1.5 },  // Mid
    { freq: 4000, gain: 0, q: 2.0 },  // High-Mid
    { freq: 10000, gain: 0, q: 1.0 }  // High
  ])

  // Backend connection state
  const [connected, setConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState(null)
  const [lastEventTimestamp, setLastEventTimestamp] = useState(0)
  const [isUpdatingFromBackend, setIsUpdatingFromBackend] = useState(false)

  const handleEffectChange = (effect, value) => {
    setEffectValues((prev) => ({
      ...prev,
      [effect]: value
    }))
  }

  const handleEqChange = async (bandIndex, value) => {
    if (isUpdatingFromBackend) return // Prevent feedback loop
    
    // Update local state immediately for responsive UI
    setEqValues((prev) => {
      const newValues = [...prev]
      newValues[bandIndex] = { ...newValues[bandIndex], gain: value }
      return newValues
    })

    // Send to backend if connected
    if (connected) {
      try {
        const result = await apiService.updateEQBand(bandIndex, 'gain_db', value)
        if (!result.success) {
          console.error('Failed to update EQ band:', result.error)
        }
      } catch (error) {
        console.error('Error updating EQ band:', error)
      }
    }
  }

  // Load EQ values from backend
  const loadEQFromBackend = async () => {
    console.log('🎛️ Loading EQ from backend...')
    try {
      const result = await apiService.getEQBands()
      console.log('🎛️ EQ bands result:', result)
      if (result.success && result.data) {
        console.log('📊 Setting EQ values:', result.data)
        setIsUpdatingFromBackend(true)
        setEqValues(result.data.map(band => ({
          freq: band.freq,
          gain: band.gain,
          q: band.q
        })))
        setTimeout(() => setIsUpdatingFromBackend(false), 100)
      } else {
        console.log('❌ Failed to load EQ bands:', result.error)
      }
    } catch (error) {
      console.error('💥 Error loading EQ from backend:', error)
    }
  }

  // Check backend connection
  const checkConnection = async () => {
    console.log('🔍 Checking backend connection...')
    try {
      const health = await apiService.healthCheck()
      console.log('🏥 Health check result:', health)
      setConnected(health.success)
      if (health.success) {
        console.log('✅ Backend connected successfully')
        // Load initial EQ values and system status
        await loadEQFromBackend()
        await loadSystemStatus()
      } else {
        console.log('❌ Backend connection failed:', health.error)
      }
    } catch (error) {
      setConnected(false)
      console.error('💥 Backend connection error:', error)
    }
  }

  // Load system status from backend
  const loadSystemStatus = async () => {
    try {
      const result = await apiService.getStatus()
      if (result.success) {
        setSystemStatus(result.data)
      }
    } catch (error) {
      console.error('Error loading system status:', error)
    }
  }

  // Poll for backend updates
  const pollBackendEvents = async () => {
    if (!connected) return

    try {
      // Get events and system status
      const [eventResult, statusResult] = await Promise.all([
        apiService.getEvents(lastEventTimestamp),
        apiService.getStatus()
      ])

      // Update system status
      if (statusResult.success) {
        setSystemStatus(statusResult.data)
      }

      // Process events
      if (eventResult.success && eventResult.data.events.length > 0) {
        const events = eventResult.data.events
        
        // Look for EQ update events
        const eqEvents = events.filter(event => event.type === 'eq_updated')
        if (eqEvents.length > 0) {
          const latestEqEvent = eqEvents[eqEvents.length - 1]
          if (latestEqEvent.data.bands) {
            setIsUpdatingFromBackend(true)
            setEqValues(latestEqEvent.data.bands.map(band => ({
              freq: band.freq,
              gain: band.gain,
              q: band.q
            })))
            setTimeout(() => setIsUpdatingFromBackend(false), 100)
          }
        }

        setLastEventTimestamp(eventResult.data.server_time)
      }
    } catch (error) {
      console.error('Error polling backend events:', error)
    }
  }

  // Reset EQ to flat
  const resetEQ = async () => {
    if (connected) {
      try {
        const result = await apiService.resetEQ()
        if (result.success) {
          await loadEQFromBackend()
        }
      } catch (error) {
        console.error('Error resetting EQ:', error)
      }
    } else {
      // Reset locally if not connected
      setEqValues(prev => prev.map(band => ({ ...band, gain: 0 })))
    }
  }

  // Start/stop audio
  const toggleAudio = async () => {
    if (!connected) return
    
    try {
      if (systemStatus?.audio_running) {
        await apiService.stopAudio()
      } else {
        await apiService.startAudio()
      }
    } catch (error) {
      console.error('Error toggling audio:', error)
    }
  }

  // Start/stop auto-EQ
  const toggleAutoEQ = async () => {
    if (!connected) return
    
    try {
      if (systemStatus?.auto_eq_running) {
        await apiService.stopAutoEQ()
      } else {
        await apiService.startAutoEQ()
      }
    } catch (error) {
      console.error('Error toggling auto-EQ:', error)
    }
  }

  // Simple camera setup
  const videoRef = useRef(null)

  useEffect(() => {
    const startCamera = () => {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream
          }
        })
        .catch((err) => {
          console.log('Camera error:', err)
        })
    }

    startCamera()

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  // Initialize backend connection and polling
  useEffect(() => {
    console.log('🚀 LiveSettingPage: Initializing backend connection...')
    checkConnection()
    
    // Set up polling interval
    const pollingInterval = setInterval(() => {
      pollBackendEvents()
    }, 1000) // Poll every second
    
    // Set up connection check interval
    const connectionInterval = setInterval(() => {
      checkConnection()
    }, 10000) // Check connection every 10 seconds
    
    return () => {
      console.log('🧹 LiveSettingPage: Cleaning up intervals...')
      clearInterval(pollingInterval)
      clearInterval(connectionInterval)
    }
  }, [connected, lastEventTimestamp])

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
        
        * {
          font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', 'Courier New', monospace !important;
        }
        
        @keyframes pop-in {
          0% { transform: scale(0.95); opacity: 0.7; }
          70% { transform: scale(1.1); opacity: 1; }
          100% { transform: scale(1); opacity: 1; }
        }
        .animate-pop-in {
          animation: pop-in 0.25s ease-out;
        }
      `}</style>
      <div className="w-screen h-screen bg-gradient-to-br from-gray-50 to-gray-100 relative overflow-hidden">
        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-15"
          style={{
            backgroundImage: `
            linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px)
          `,
            backgroundSize: '20px 20px'
          }}
        />

        {/* Back button - positioned absolutely */}
        <button
          onClick={onBack}
          className="absolute top-6 left-6 z-50 flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200 cursor-pointer p-2 rounded"
          style={{ zIndex: 9999 }}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
          <span className="font-medium">Back</span>
        </button>

        {/* Main Layout - Two Column Structure */}
        <div className="relative z-10 w-full h-full flex p-4 pt-16">
          {/* Left Column - Much Wider */}
          <div className="flex-[3] flex flex-col mr-4">
            {/* Top Left - Live Audio Visualizer */}
            <div className="flex-1 bg-black rounded-lg shadow-lg border border-gray-200 mb-2 overflow-hidden">
              <FluidVisualizer />
            </div>

            {/* Bottom Left - Live EQ Control */}
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-bold text-gray-800">Live EQ Control</h3>
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    connected ? 'bg-green-500' : 'bg-red-500'
                  }`} title={connected ? 'Backend Connected' : 'Backend Disconnected'} />
                  <button
                    onClick={resetEQ}
                    className="px-3 py-1 bg-gray-500 hover:bg-gray-600 text-white text-sm rounded transition-colors"
                    disabled={!connected}
                  >
                    Reset
                  </button>
                </div>
              </div>
              
              <div className="flex justify-center space-x-4 mb-4">
                {/* 5 EQ Sliders matching backend bands */}
                {eqValues.map((band, index) => (
                  <Slider
                    key={index}
                    initialValue={band.gain}
                    min={-12}
                    max={12}
                    onChange={(value) => handleEqChange(index, value)}
                    label={`${band.freq >= 1000 ? (band.freq / 1000).toFixed(0) + 'k' : band.freq}Hz`}
                    disabled={!connected}
                  />
                ))}
              </div>
              
              <div className="flex justify-center space-x-2">
                <button
                  onClick={toggleAudio}
                  className={`px-4 py-2 rounded transition-colors ${
                    systemStatus?.audio_running
                      ? 'bg-red-500 hover:bg-red-600 text-white'
                      : 'bg-green-500 hover:bg-green-600 text-white'
                  }`}
                  disabled={!connected}
                >
                  {systemStatus?.audio_running ? 'Stop Audio' : 'Start Audio'}
                </button>
                <button
                  onClick={toggleAutoEQ}
                  className={`px-4 py-2 rounded transition-colors ${
                    systemStatus?.auto_eq_running
                      ? 'bg-orange-500 hover:bg-orange-600 text-white'
                      : 'bg-blue-500 hover:bg-blue-600 text-white'
                  }`}
                  disabled={!connected || !systemStatus?.audio_running}
                >
                  {systemStatus?.auto_eq_running ? 'Stop Auto-EQ' : 'Start Auto-EQ'}
                </button>
              </div>
            </div>
          </div>

          {/* Right Column - Narrower */}
          <div className="flex-[1] flex flex-col space-y-2">
            {/* Top Right - Effects Knobs */}
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
              <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">
                Effects (Manual Override)
              </h3>
              <div className="grid grid-cols-3 gap-2 gap-y-2">
                <Knob
                  label="Reverb"
                  value={effectValues.reverb}
                  onChange={(value) => handleEffectChange('reverb', value)}
                />
                <Knob
                  label="Delay"
                  value={effectValues.delay}
                  onChange={(value) => handleEffectChange('delay', value)}
                />
                <Knob
                  label="Overdrive"
                  value={effectValues.overdrive}
                  onChange={(value) => handleEffectChange('overdrive', value)}
                />
                <Knob
                  label="Fuzz"
                  value={effectValues.fuzz}
                  onChange={(value) => handleEffectChange('fuzz', value)}
                />
                <Knob
                  label="Compression"
                  value={effectValues.compression}
                  onChange={(value) => handleEffectChange('compression', value)}
                />
                <Knob
                  label="Gain"
                  value={effectValues.gain}
                  onChange={(value) => handleEffectChange('gain', value)}
                />
              </div>
            </div>

            {/* Bottom Right - Video Section */}
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
              <h3 className="text-lg font-bold text-gray-800 mb-2">Live Camera</h3>
              <div className="w-full aspect-video bg-gray-100 rounded border border-gray-300 overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                  style={{ transform: 'scaleX(-1)' }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default LiveSettingPage
