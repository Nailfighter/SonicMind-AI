import React from 'react'
import ProfileCard from './ProfileCard.jsx'

const ProfileSelector = ({ profiles, onProfileSelect, onCreateNew }) => {
  return (
    <div className="flex flex-col items-center space-y-8">
      <div className="text-center">
        <h1 className="text-8xl font-bold mb-4">
          <span className="text-black font-black">SONIC</span>
          <span className="bg-gradient-to-r from-cyan-600 via-blue-700 via-purple-800 via-pink-700 to-orange-600 bg-clip-text text-transparent">
            MINDS
          </span>
        </h1>
        <p className="text-gray-800 text-xl font-semibold">AI Assistants for Sound Engineers</p>
      </div>

      <div className="flex space-x-8 items-center">
        {profiles.map((profile, index) => (
          <ProfileCard
            key={profile.id}
            profile={profile}
            onClick={() => onProfileSelect(profile)}
          />
        ))}
        <ProfileCard isNew={true} onClick={onCreateNew} />
      </div>
    </div>
  )
}

export default ProfileSelector
