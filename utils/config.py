"""
⚙️ Configuration cho environments (REMOVED device configs)
"""

# Chỉ giữ ENVIRONMENT_CONFIGS, loại bỏ OUTPUT_DEVICE_CONFIGS
ENVIRONMENT_CONFIGS = {
    
    'airport': {
        'name': 'Sân bay',
        'description': 'Jet engine + thông báo + đám đông + hành lý',
        'color': '#C1C0C0',
        'icon': ''
    },
    
    'cafe': {
        'name': 'Quán cà phê',
        'description': 'Máy pha cà phê + tiếng nói + chén dĩa + nhạc nền',
        'color': '#C1C0C0',
        'icon': ''
    },
    
    'office': {
        'name': 'Văn phòng',
        'description': 'AC hum + keyboard typing + máy in + điện thoại',
        'color': '#C1C0C0',
        'icon': ''
    },
    'home': {
        'name': 'Nhà riêng',
        'description': 'TV background + thiết bị gia dụng + rất ít nhiễu',
        'color': '#C1C0C0',
        'icon': ''
    }
}

def get_environment_by_name(name: str) -> dict:
    """Tìm environment config theo tên"""
    for env_id, config in ENVIRONMENT_CONFIGS.items():
        if config['name'].lower() == name.lower():
            return {env_id: config}
    return {}


