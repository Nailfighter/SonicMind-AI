import React from 'react'

const ProfileCard = ({ profile, isNew = false, onClick }) => {
  if (isNew) {
    return (
      <div 
        className="flex flex-col items-center justify-center cursor-pointer transition-all duration-300"
        onClick={onClick}
      >
        <div className="w-32 h-32 rounded-full border-2 border-dashed border-blue-400 hover:border-blue-600 transition-all duration-300 bg-gradient-to-br from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 hover:shadow-lg hover:shadow-blue-200/50 flex items-center justify-center">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center shadow-md">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
          </div>
        </div>
        <span className="text-sm text-gray-700 font-semibold mt-3">New Profile</span>
      </div>
    )
  }

  return (
    <div 
      className="flex flex-col items-center justify-center cursor-pointer transition-all duration-200"
      onClick={onClick}
    >
      {/* Music disk container */}
      <div className="relative w-32 h-32 rounded-full shadow-lg hover:shadow-xl transition-all duration-200 overflow-hidden">
        {/* Music disk outer ring */}
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-gray-800 via-gray-700 to-gray-900"></div>
        
        {/* Inner disk with grooves */}
        <div className="absolute inset-2 rounded-full bg-gradient-to-br from-gray-600 to-gray-800 flex items-center justify-center">
          {/* Center hole */}
          <div className="w-8 h-8 rounded-full bg-gray-900 flex items-center justify-center">
            <div className="w-4 h-4 rounded-full bg-gray-700"></div>
          </div>
        </div>
        
        {/* Profile initial - centered on disk */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-16 h-16 rounded-full bg-white/90 flex items-center justify-center shadow-inner">
            <span className="text-2xl font-bold text-gray-800">
              {profile.name.charAt(0).toUpperCase()}
            </span>
          </div>
        </div>
      </div>
      
      {/* Profile name below disk */}
      <span className="text-sm text-gray-800 font-medium mt-3">{profile.name}</span>
    </div>
  )
}

export default ProfileCard
