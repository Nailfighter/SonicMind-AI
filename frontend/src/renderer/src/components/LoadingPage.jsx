import React, { useEffect, useState } from 'react'

const LoadingPage = ({ onComplete }) => {
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState(0)

  const steps = [
    'Detecting environment...',
    'Detecting material properties...',
    'Adjusting preset...',
    'Detecting audience levels...',
    'Detecting instrument...',
    'Detecting relative position of artist...',
    'Using SonicMind AI...',
    'Ready to go!'
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setTimeout(() => onComplete(), 1000)
          return 100
        }
        return prev + 1.5
      })
    }, 100)

    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= steps.length - 1) {
          clearInterval(stepInterval)
          return prev
        }
        return prev + 1
      })
    }, 1500)

    return () => {
      clearInterval(interval)
      clearInterval(stepInterval)
    }
  }, [onComplete])

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
      
      {/* Main content */}
      <div className="relative z-10 w-full h-full flex flex-col items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Logo/Icon */}
          <div className="text-center mb-8">
            {/* Moving Waveform Animation */}
            <div className="w-32 h-32 bg-gradient-to-br from-emerald-500 to-orange-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-xl">
              <div className="w-24 h-24 flex items-center justify-center">
                <svg className="w-20 h-20 text-white" viewBox="0 0 120 50" fill="none">
                  {/* Animated waveform bars - Bigger and more dynamic */}
                  <rect x="5" y="20" width="4" height="10" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0s' }} />
                  <rect x="12" y="15" width="4" height="20" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.1s' }} />
                  <rect x="19" y="8" width="4" height="34" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.2s' }} />
                  <rect x="26" y="12" width="4" height="26" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.3s' }} />
                  <rect x="33" y="18" width="4" height="14" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.4s' }} />
                  <rect x="40" y="25" width="4" height="0" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.5s' }} />
                  <rect x="47" y="22" width="4" height="6" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.6s' }} />
                  <rect x="54" y="10" width="4" height="30" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.7s' }} />
                  <rect x="61" y="14" width="4" height="22" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.8s' }} />
                  <rect x="68" y="20" width="4" height="10" fill="currentColor" className="animate-pulse" style={{ animationDelay: '0.9s' }} />
                  <rect x="75" y="12" width="4" height="26" fill="currentColor" className="animate-pulse" style={{ animationDelay: '1s' }} />
                  <rect x="82" y="16" width="4" height="18" fill="currentColor" className="animate-pulse" style={{ animationDelay: '1.1s' }} />
                  <rect x="89" y="22" width="4" height="6" fill="currentColor" className="animate-pulse" style={{ animationDelay: '1.2s' }} />
                  <rect x="96" y="15" width="4" height="20" fill="currentColor" className="animate-pulse" style={{ animationDelay: '1.3s' }} />
                  <rect x="103" y="10" width="4" height="30" fill="currentColor" className="animate-pulse" style={{ animationDelay: '1.4s' }} />
                  <rect x="110" y="18" width="4" height="14" fill="currentColor" className="animate-pulse" style={{ animationDelay: '1.5s' }} />
                </svg>
              </div>
            </div>
            
            {/* SonicMind Branding */}
            <h1 className="text-4xl font-bold mb-2">
              <span className="text-black font-black">SONIC</span>
              <span className="bg-gradient-to-r from-cyan-600 via-blue-700 via-purple-800 via-pink-700 to-orange-600 bg-clip-text text-transparent">
                MINDS
              </span>
            </h1>
            <p className="text-gray-600 text-lg">Preparing your live session</p>
          </div>

          {/* Progress bar */}
          <div className="mb-8">
            <div className="w-full bg-gray-200 rounded-full h-3 mb-3 shadow-inner">
              <div 
                className="bg-gradient-to-r from-emerald-500 via-teal-600 via-amber-500 to-orange-500 h-3 rounded-full transition-all duration-500 ease-out shadow-lg"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <div className="text-center text-lg font-semibold text-gray-700">
              {progress}%
            </div>
          </div>

          {/* Current step - Emphasized */}
          <div className="text-center bg-white rounded-2xl p-6 shadow-xl border-2 border-gray-100 mb-6">
            <div className="text-2xl font-bold text-gray-800 mb-4 min-h-[3rem] flex items-center justify-center">
              {steps[currentStep]}
            </div>
            <div className="flex justify-center space-x-2 mb-4">
              {steps.map((_, index) => (
                <div
                  key={index}
                  className={`w-5 h-5 rounded-full transition-all duration-500 ${
                    index < currentStep 
                      ? 'bg-emerald-500 scale-125 shadow-lg ring-2 ring-emerald-200' 
                      : index === currentStep 
                        ? 'bg-orange-500 animate-pulse scale-110 shadow-lg ring-2 ring-orange-200' 
                        : 'bg-gray-300'
                  }`}
                />
              ))}
            </div>
            <div className="text-lg font-bold text-gray-700">
              Step {currentStep + 1} of {steps.length}
            </div>
          </div>

          {/* Animated dots */}
          <div className="flex justify-center mt-8 space-x-2">
            <div className="w-3 h-3 bg-emerald-500 rounded-full animate-bounce shadow-md"></div>
            <div className="w-3 h-3 bg-teal-500 rounded-full animate-bounce shadow-md" style={{ animationDelay: '0.2s' }}></div>
            <div className="w-3 h-3 bg-orange-500 rounded-full animate-bounce shadow-md" style={{ animationDelay: '0.4s' }}></div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default LoadingPage
