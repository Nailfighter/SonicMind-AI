import React, { useState } from 'react'

const App = () => {
  const [rotation, setRotation] = useState(0)

  const handleWheel = (event) => {
    event.preventDefault()

    let newRotation = rotation

    if (event.deltaY > 0) {
      newRotation = Math.min(rotation + 5, 270)
    } else {
      newRotation = Math.max(rotation - 5, 0)
    }

    setRotation(newRotation)
  }

  const startAngle = 225
  const knobRotation = startAngle + rotation

  return (
      <div className="relative w-40 h-40" onWheel={handleWheel}>
        <div className="absolute inset-0 rounded-full p-[6px]" style={{}}>
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: `conic-gradient(from 225deg, #fcd34d 0deg, #fcd34d ${rotation}deg, transparent ${rotation}deg)`
            }}
          ></div>

          <div
            className="absolute inset-[6px] rounded-full bg-neutral-800 flex items-center justify-center"
            style={{}}
          >
            <div
              className="absolute inset-0 transition-transform duration-100 ease-linear"
              style={{
                transform: `rotate(${knobRotation}deg)`
              }}
            >
              <div
                className="w-4 h-6 rounded-full bg-amber-400 absolute left-1/2 -translate-x-1/2"
                style={{ top: '10px' }}
              ></div>
            </div>
          </div>
        </div>
      </div>
  )
}

export default App
