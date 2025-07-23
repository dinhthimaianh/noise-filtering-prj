"""
⚙️ Configuration cho environments (REMOVED device configs)
"""

# Chỉ giữ ENVIRONMENT_CONFIGS, loại bỏ OUTPUT_DEVICE_CONFIGS
ENVIRONMENT_CONFIGS = {
    
    'airport': {
        'name': 'Sân bay',
        'description': 'Jet engine + thông báo + đám đông + hành lý',
        'target_snr_db': 0,
        'dominant_frequencies': [100, 200, 500, 1000],
        'characteristics': [
            'broadband_noise',
            'low_frequency_rumble',
            'announcements',
            'crowd_noise',
            'rolling_luggage'
        ],
        'processing_strategy': 'aggressive',
        'priority_bands': {
            'speech_protection': [500, 4000],
            'noise_suppression': [50, 100, 200, 400, 800]
        },
        'color': '#C1C0C0',
        'icon': ''
    },
    
    'cafe': {
        'name': 'Quán cà phê',
        'description': 'Máy pha cà phê + tiếng nói + chén dĩa + nhạc nền',
        'target_snr_db': 5,
        'dominant_frequencies': [200, 400, 800, 2000],
        'characteristics': [
            'coffee_machine',
            'human_chatter',
            'dishes_clinking',
            'background_music',
            'steam_noise'
        ],
        'processing_strategy': 'moderate',
        'priority_bands': {
            'speech_protection': [400, 3000],
            'noise_suppression': [200, 500, 1000, 2000, 4000]
        },
        'color': '#C1C0C0',
        'icon': ''
    },
    
    'office': {
        'name': 'Văn phòng',
        'description': 'AC hum + keyboard typing + máy in + điện thoại',
        'target_snr_db': 15,
        'dominant_frequencies': [60, 120, 500, 4000],
        'characteristics': [
            'ac_hum',
            'keyboard_typing',
            'printer_noise',
            'phone_rings',
            'paper_rustling'
        ],
        'processing_strategy': 'minimal',
        'priority_bands': {
            'speech_protection': [300, 3400],
            'noise_suppression': [60, 120, 500]
        },
        'color': '#C1C0C0',
        'icon': ''
    },
    'home': {
        'name': 'Nhà riêng',
        'description': 'TV background + thiết bị gia dụng + rất ít nhiễu',
        'target_snr_db': 20,
        'dominant_frequencies': [60, 120, 1000],
        'characteristics': [
            'tv_background',
            'appliance_hums',
            'minimal_noise',
            'fridge_hum',
            'subtle_air_conditioning'
        ],
        'processing_strategy': 'natural',
        'priority_bands': {
            'speech_protection': [200, 4000],
            'noise_suppression': [60, 120]
        },
        'color': '#C1C0C0',
        'icon': ''
    }
}



# Performance benchmarks (giữ nguyên)
PERFORMANCE_TARGETS = {
    'real_time_factor_min': 2.0,
    'latency_max_ms': 50,
    'snr_improvement_min_db': 3,
    'thd_max_percent': 1.0,
    'memory_usage_max_mb': 256,
    'cpu_usage_max_percent': 80
}

def get_environment_by_name(name: str) -> dict:
    """Tìm environment config theo tên"""
    for env_id, config in ENVIRONMENT_CONFIGS.items():
        if config['name'].lower() == name.lower():
            return {env_id: config}
    return {}


