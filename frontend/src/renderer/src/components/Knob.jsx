import React, { useState } from 'react'

const Knob = ({ value = 0, onChange, label, maxValue = 270 }) => {
  const [rotation, setRotation] = useState(value)

  const handleWheel = (event) => {
    event.preventDefault()

    let newRotation = rotation

    if (event.deltaY > 0) {
      newRotation = Math.min(rotation + 5, maxValue)
    } else {
      newRotation = Math.max(rotation - 5, 0)
    }

    setRotation(newRotation)
    if (onChange) {
      onChange(newRotation)
    }
  }

  const startAngle = 225
  const knobRotation = startAngle + rotation

  return (
    <div className="flex flex-col items-center space-y-1">
      {label && (
        <div className="bg-gray-100 px-2 py-1 rounded-md border border-gray-200 shadow-sm">
          <span className="text-xs font-semibold text-gray-800 text-center tracking-wide uppercase">
            {label}
          </span>
        </div>
      )}
      <div className="relative w-16 h-16" onWheel={handleWheel}>
        <div className="absolute inset-0 rounded-full p-[2px] shadow-inner">
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: `conic-gradient(from 225deg, #fcd34d 0deg, #fcd34d ${rotation}deg, transparent ${rotation}deg)`
            }}
          ></div>

          <div className="absolute inset-[2px] rounded-full bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center shadow-lg border border-gray-600">
            <div
              className="absolute inset-0 transition-transform duration-100 ease-linear"
              style={{
                transform: `rotate(${knobRotation}deg)`
              }}
            >
              <div
                className="w-1 h-2 rounded-full bg-amber-400 absolute left-1/2 -translate-x-1/2 shadow-sm"
                style={{ top: '3px' }}
              ></div>
            </div>
          </div>
        </div>
      </div>
      <div className="bg-gray-50 px-1.5 py-0.5 rounded border border-gray-200">
        <span className="text-xs text-gray-600 font-mono font-medium">
          {Math.round((rotation / maxValue) * 100)}
        </span>
      </div>
    </div>
  )
}

export default Knob
