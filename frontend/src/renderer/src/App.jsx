import React, { useState } from 'react'
import "./assets/main.css"
export default function App() {
  const [bands, setBands] = useState({
    bass: 50,
    mid: 50,
    treble: 50
  })

  const handleChange = (e) => {
    setBands({
      ...bands,
      [e.target.name]: e.target.value
    })
  }

  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center bg-gray-900 text-white font-sans">
      <h1 className="text-3xl font-bold mb-8">🎵 Music Equalizer Dashboard</h1>

      <div className="grid grid-cols-3 gap-8">
        {Object.entries(bands).map(([band, value]) => (
          <div key={band} className="flex flex-col items-center">
            <label className="capitalize text-lg mb-2">{band}</label>
            <input
              type="range"
              name={band}
              min="0"
              max="100"
              value={value}
              onChange={handleChange}
              className="w-32 accent-green-400"
            />
            <span className="mt-2">{value}</span>
          </div>
        ))}
      </div>

      <div className="flex items-end gap-1 mt-10 h-40">
        {Object.values(bands).map((value, idx) => (
          <div
            key={idx}
            className="w-8 bg-gradient-to-t from-green-600 to-green-300 rounded-md transition-all duration-300"
            style={{ height: `${value * 1.5}px` }}
          ></div>
        ))}
      </div>
    </div>
  )
}
