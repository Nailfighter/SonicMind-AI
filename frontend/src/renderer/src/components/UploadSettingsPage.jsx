import React, { useState } from 'react'
import HandleSvg from '../assets/Handle.svg'

const UploadSettingsPage = ({ profile, uploadData, onBack, onContinue }) => {
  const [videoSettings, setVideoSettings] = useState({
    effects: [false, false, false, false],
    audio: [false, false, false, false, false]
  })

  const handleEffectToggle = (index) => {
    setVideoSettings(prev => ({
      ...prev,
      effects: prev.effects.map((item, i) => i === index ? !item : item)
    }))
  }

  const handleAudioToggle = (index) => {
    setVideoSettings(prev => ({
      ...prev,
      audio: prev.audio.map((item, i) => i === index ? !item : item)
    }))
  }

  const handleSliderChange = (type, index, value) => {
    // Handle slider value changes
    console.log(`${type} slider ${index} changed to:`, value)
  }

  return (
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
      
      {/* Back button */}
      <button 
        onClick={onBack}
        className="absolute top-6 left-6 z-50 flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200 cursor-pointer p-2 rounded"
        style={{ zIndex: 9999 }}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        <span className="font-medium">Back</span>
      </button>

      {/* Header */}
      <div className="relative z-10 flex justify-end items-center p-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-800 via-gray-700 to-gray-900 flex items-center justify-center shadow-lg">
            <span className="text-lg font-bold text-white">
              {profile.name.charAt(0).toUpperCase()}
            </span>
          </div>
          <div className="text-right">
            <h3 className="text-lg font-bold text-gray-800">{profile.name}</h3>
            <p className="text-sm text-gray-600">
              {uploadData?.instrument ? `${uploadData.instrument.name} Processing` : 'Upload Mode'}
            </p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10 flex flex-col h-full px-6 pt-4 pb-6" style={{ zIndex: 1 }}>
        <div className="flex-1 flex space-x-4">
          
          {/* Left Side - WAVE Section */}
          <div className="flex-1 bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">WAVE</h2>
            <div className="w-full h-64 bg-gray-100 rounded border border-gray-300 flex items-center justify-center relative">
              {/* Waveform visualization */}
              <svg className="w-full h-32 text-gray-400" viewBox="0 0 400 120" fill="none">
                <path
                  d="M20,60 Q60,20 100,60 T180,60 T260,60 T340,60 T380,60"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  className="animate-pulse"
                />
                <path
                  d="M20,70 Q60,90 100,70 T180,70 T260,70 T340,70 T380,70"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  fill="none"
                  className="animate-pulse"
                  style={{ animationDelay: '0.5s' }}
                />
              </svg>
              {/* Timeline indicator */}
              <div className="absolute bottom-2 left-4 right-4 h-1 bg-gray-300 rounded-full">
                <div className="h-full bg-blue-500 rounded-full w-1/3 relative">
                  <div className="absolute right-0 top-1/2 transform -translate-y-1/2 w-2 h-2 bg-blue-500 rounded-full"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Handle Asset - Vertical Separator */}
          <div className="flex items-center justify-center">
            <div className="w-8 h-20 bg-gray-200 rounded-full flex items-center justify-center shadow-lg">
              <img 
                src={HandleSvg} 
                alt="Handle" 
                className="w-6 h-16 object-contain"
              />
            </div>
          </div>

          {/* Right Side - VIDEO Section */}
          <div className="flex-1 bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-800">VIDEO</h2>
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <span>orig</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
            <div className="w-full h-64 bg-gray-100 rounded border border-gray-300 flex items-center justify-center">
              {uploadData?.file ? (
                <div className="w-full h-full flex items-center justify-center">
                  <video 
                    className="w-full h-full object-cover rounded"
                    controls
                    src={URL.createObjectURL(uploadData.file)}
                  />
                </div>
              ) : (
                <div className="text-center">
                  <svg className="w-16 h-16 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <p className="text-gray-500">Original Video Preview</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Control Panels */}
        <div className="flex space-x-6 mt-4">
          {/* Left Control Panel - Effects */}
          <div className="flex-1 bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Effects</h3>
            <div className="space-y-4">
              {videoSettings.effects.map((isActive, index) => (
                <div key={index} className="flex items-center space-x-4">
                  <input
                    type="checkbox"
                    checked={isActive}
                    onChange={() => handleEffectToggle(index)}
                    className="w-5 h-5 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <div className="flex-1 flex items-center space-x-3">
                    {/* Handle Asset instead of slider */}
                    <div className="flex-1 flex items-center justify-center">
                      <div className="w-8 h-16 bg-gray-200 rounded-full flex items-center justify-center shadow-lg">
                        <img 
                          src={HandleSvg} 
                          alt="Handle" 
                          className="w-6 h-12 object-contain"
                        />
                      </div>
                    </div>
                    <span className="text-sm text-gray-600 w-8">Fx{index + 1}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Control Panel - Audio */}
          <div className="flex-1 bg-white rounded-lg shadow-lg border border-gray-200 p-6">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Audio</h3>
            <div className="space-y-4">
              {videoSettings.audio.map((isActive, index) => (
                <div key={index} className="flex items-center space-x-4">
                  <input
                    type="checkbox"
                    checked={isActive}
                    onChange={() => handleAudioToggle(index)}
                    className="w-5 h-5 text-green-600 bg-gray-100 border-gray-300 rounded focus:ring-green-500"
                  />
                  <div className="flex-1 flex items-center space-x-3">
                    {/* Handle Asset instead of slider */}
                    <div className="flex-1 flex items-center justify-center">
                      <div className="w-8 h-16 bg-gray-200 rounded-full flex items-center justify-center shadow-lg">
                        <img 
                          src={HandleSvg} 
                          alt="Handle" 
                          className="w-6 h-12 object-contain"
                        />
                      </div>
                    </div>
                    <span className="text-sm text-gray-600 w-8">A{index + 1}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Continue Button */}
        <div className="text-center mt-6">
          <button
            onClick={onContinue}
            className="bg-gray-800 text-white px-12 py-4 rounded-full font-semibold text-lg hover:bg-gray-700 transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            Process Video
          </button>
        </div>
      </div>
    </div>
  )
}

export default UploadSettingsPage
