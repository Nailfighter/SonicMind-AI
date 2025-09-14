import React, { useState, useRef, useEffect } from 'react'
import Knob from './Knob'
import Slider from './Slider'
import FluidVisualizer from './FluidVisualizer'

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

  // State for EQ slider values
  const [eqValues, setEqValues] = useState({
    low: 0,
    lowMid: 0,
    mid: 0,
    highMid: 0,
    high: 0
  })

  const handleEffectChange = (effect, value) => {
    setEffectValues((prev) => ({
      ...prev,
      [effect]: value
    }))
  }

  const handleEqChange = (band, value) => {
    setEqValues((prev) => ({
      ...prev,
      [band]: value
    }))
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
              <h3 className="text-lg font-bold text-gray-800 text-center mb-4">Live EQ Control</h3>
              <div className="flex justify-center space-x-4">
                {/* 5 EQ Sliders */}
                <Slider
                  initialValue={eqValues.low}
                  min={-20}
                  max={20}
                  onChange={(value) => handleEqChange('low', value)}
                  label="80Hz"
                />
                <Slider
                  initialValue={eqValues.lowMid}
                  min={-20}
                  max={20}
                  onChange={(value) => handleEqChange('lowMid', value)}
                  label="250Hz"
                />
                <Slider
                  initialValue={eqValues.mid}
                  min={-20}
                  max={20}
                  onChange={(value) => handleEqChange('mid', value)}
                  label="1kHz"
                />
                <Slider
                  initialValue={eqValues.highMid}
                  min={-20}
                  max={20}
                  onChange={(value) => handleEqChange('highMid', value)}
                  label="4kHz"
                />
                <Slider
                  initialValue={eqValues.high}
                  min={-20}
                  max={20}
                  onChange={(value) => handleEqChange('high', value)}
                  label="12kHz"
                />
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
