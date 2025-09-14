// Simple Node.js script to test backend connection
const fetch = require('node-fetch');

const API_BASE = 'http://localhost:8001';

async function testConnection() {
    console.log('Testing SonicMind AI Backend Connection...\n');
    
    try {
        // Test health check
        console.log('1. Testing health check...');
        const healthResponse = await fetch(`${API_BASE}/api/health`);
        const healthData = await healthResponse.json();
        console.log('✅ Health check:', healthData.status);
        console.log('   Systems:', JSON.stringify(healthData.systems, null, 2));
        
        // Test EQ bands
        console.log('\n2. Testing EQ bands...');
        const eqResponse = await fetch(`${API_BASE}/api/eq/bands`);
        const eqData = await eqResponse.json();
        console.log('✅ EQ bands retrieved:', eqData.bands.length, 'bands');
        console.log('   Bands:', eqData.bands.map(b => `${b.freq}Hz: ${b.gain}dB`).join(', '));
        
        // Test updating EQ band
        console.log('\n3. Testing EQ band update...');
        const updateResponse = await fetch(`${API_BASE}/api/eq/bands/0`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ parameter: 'gain_db', value: 2.5 })
        });
        const updateData = await updateResponse.json();
        console.log('✅ EQ update result:', updateData.message);
        
        // Test reset
        console.log('\n4. Testing EQ reset...');
        const resetResponse = await fetch(`${API_BASE}/api/eq/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const resetData = await resetResponse.json();
        console.log('✅ EQ reset result:', resetData.message);
        
        console.log('\n🎉 All backend API tests passed!');
        console.log('\nNow you can run the frontend with: npm run dev');
        
    } catch (error) {
        console.error('❌ Backend connection failed:', error.message);
        console.log('\nMake sure the backend server is running:');
        console.log('python backend/rest_api_server.py');
    }
}

testConnection();