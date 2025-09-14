import React from 'react'

const LiveSettingPage = ({ profile, deviceSettings, onBack }) => {
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
      
      {/* Window controls - positioned absolutely at top right */}
      <div className="absolute top-4 right-4 z-50 flex space-x-2">
        <button className="w-6 h-6 rounded-full bg-gray-300 hover:bg-gray-400 transition-colors duration-200 flex items-center justify-center">
          <span className="text-gray-600 text-xs font-bold">−</span>
        </button>
        <button className="w-6 h-6 rounded-full bg-gray-300 hover:bg-gray-400 transition-colors duration-200 flex items-center justify-center">
          <span className="text-gray-600 text-xs font-bold">□</span>
        </button>
        <button className="w-6 h-6 rounded-full bg-gray-300 hover:bg-gray-400 transition-colors duration-200 flex items-center justify-center">
          <span className="text-gray-600 text-xs font-bold">×</span>
        </button>
      </div>

      {/* Back button - positioned absolutely */}
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

      {/* Main Layout - Two Column Structure */}
      <div className="relative z-10 w-full h-full flex p-4 pt-16">
        {/* Left Column - Wider */}
        <div className="flex-1 flex flex-col mr-2">
          {/* Top Left - Waveform Display */}
          <div className="flex-1 bg-white rounded-lg shadow-lg border border-gray-200 mb-2 flex items-center justify-center">
            <div className="w-full h-full flex items-center justify-center">
              {/* Waveform placeholder */}
              <div className="w-full h-32 flex items-center justify-center">
                <svg className="w-full h-20 text-gray-300" viewBox="0 0 400 80" fill="none">
                  <path
                    d="M20,40 Q60,10 100,40 T180,40 T260,40 T340,40 T380,40"
                    stroke="currentColor"
                    strokeWidth="2"
                    fill="none"
                    className="animate-pulse"
                  />
                  <path
                    d="M20,50 Q60,70 100,50 T180,50 T260,50 T340,50 T380,50"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    fill="none"
                    className="animate-pulse"
                    style={{ animationDelay: '0.5s' }}
                  />
                </svg>
              </div>
            </div>
          </div>

          {/* Bottom Left - Live EQ Control */}
          <div className="h-48 bg-white rounded-lg shadow-lg border border-gray-200 p-4">
            <h3 className="text-lg font-bold text-gray-800 text-center mb-4">Live EQ Control</h3>
            <div className="flex justify-center space-x-4 h-full">
              {/* 5 EQ Fader Containers */}
              {[1, 2, 3, 4, 5].map((fader) => (
                <div key={fader} className="flex flex-col items-center space-y-2">
                  <div className="w-8 h-24 bg-gray-100 rounded-lg border border-gray-300 flex flex-col justify-end relative">
                    {/* Fader knob placeholder */}
                    <div className="w-6 h-2 bg-gray-400 rounded-full mx-auto mb-2"></div>
                  </div>
                  <span className="text-xs text-gray-600 font-medium">{fader}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column - Narrower */}
        <div className="w-80 flex flex-col space-y-2">
          {/* Top Right - Effects List */}
          <div className="flex-1 bg-white rounded-lg shadow-lg border border-gray-200 p-4">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Effects</h3>
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map((effect) => (
                <div key={effect} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">Fx {effect}</span>
                  <div className="w-6 h-6 bg-gray-200 rounded-full border border-gray-300"></div>
                </div>
              ))}
            </div>
          </div>

          {/* Middle Right - Question Mark Section */}
          <div className="h-24 bg-white rounded-lg shadow-lg border border-gray-200 flex items-center justify-center">
            <span className="text-4xl text-gray-400 font-bold">?</span>
          </div>

          {/* Bottom Right - Device Settings */}
          <div className="h-32 bg-white rounded-lg shadow-lg border border-gray-200 p-4">
            <h3 className="text-lg font-bold text-gray-800 mb-2">Device Settings</h3>
            <div className="w-full h-full space-y-1">
              {deviceSettings ? (
                <div className="text-xs text-gray-600 space-y-1">
                  <div className="flex justify-between">
                    <span>Audience Camera:</span>
                    <span className="font-medium">{deviceSettings.AudienceCamera}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Artist Camera:</span>
                    <span className="font-medium">{deviceSettings.ArtistCamera}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Artist Mic:</span>
                    <span className="font-medium">{deviceSettings.ArtistMicrophone}</span>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full bg-gray-100 rounded border border-gray-300 flex items-center justify-center">
                  <span className="text-gray-400 text-sm">No device settings</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LiveSettingPage
