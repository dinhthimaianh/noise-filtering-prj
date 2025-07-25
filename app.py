"""
 Flask Web Server cho Noise Filtering Demo
Giao diện web 
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import soundfile as sf
from pydub import AudioSegment

import logging
import os
import uuid

from pathlib import Path
from datetime import datetime


from pipeline.main_pipeline import AudioProcessingPipeline
from utils.config import ENVIRONMENT_CONFIGS
from utils.audio_utils import analyze_audio_properties_fast, compute_frequency_response_fast

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = 'noise-filtering-demo-secret-key-2024'
CORS(app)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo pipeline
try:
    pipeline = AudioProcessingPipeline()
    logger.info(" Pipeline đã được khởi tạo thành công")
except Exception as e:
    logger.error(f" Lỗi khởi tạo pipeline: {e}")
    pipeline = None

# Tạo thư mục lưu trữ tạm
UPLOAD_FOLDER = Path('static/audio')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    """Trang chính - giao diện demo"""
    return render_template('index.html', 
                         environments=ENVIRONMENT_CONFIGS
                         )

@app.route('/api/environments')
def get_environments():
    """API lấy danh sách môi trường"""
    try:
        # Chuyển đổi config thành format phù hợp cho frontend
        environments = []
        for env_id, config in ENVIRONMENT_CONFIGS.items():
            environments.append({
                'id': env_id,
                'name': config['name'],
                'description': config['description'],
                'target_snr_db': config['target_snr_db'],
                'characteristics': config['characteristics']
            })
        
        return jsonify({
            'success': True,
            'environments': environments
        })
    except Exception as e:
        logger.error(f" Lỗi get environments: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """
    API xử lý audio chính - Simplified without device selection
    """
    if not pipeline:
        return jsonify({'success': False, 'error': 'Pipeline chưa được khởi tạo'}), 500
    
    try:
        # 1. Validate request
        if 'audio_file' not in request.files:
            return jsonify({'success': False, 'error': 'Không có file audio'}), 400
        
        audio_file = request.files['audio_file']
        environment = request.form.get('environment', 'office')
        # Remove: output_device = request.form.get('output_device', 'speakers')
        
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'Không có file được chọn'}), 400
        
        logger.info(f" Xử lý: {audio_file.filename}, môi trường: {environment}")
        
        # 2. Load audio có nhiễu
        try:
            temp_id = str(uuid.uuid4())
            temp_input_path = UPLOAD_FOLDER / f"input_{temp_id}.wav"
            audio_file.save(str(temp_input_path))
            
            # Load audio (đã có nhiễu)
            noisy_input, sample_rate = sf.read(str(temp_input_path))
            noisy_input_original = AudioSegment.from_file(str(temp_input_path))
            logger.info(f" Đã load audio: {audio_file.filename} ({len(noisy_input)} samples @ {sample_rate}Hz)")

            # Chuyển stereo thành mono nếu cần
            if len(noisy_input.shape) > 1:
                noisy_input = np.mean(noisy_input, axis=1)
                logger.info(" Chuyển đổi stereo thành mono")
            
            # Validate độ dài audio
            duration = len(noisy_input) / sample_rate
            if duration > 30:
                logger.warning(f"Audio quá dài ({duration:.1f}s), cắt xuống 30s")
                noisy_input = noisy_input[:30 * sample_rate]
            
            logger.info(f"Load audio có nhiễu: {len(noisy_input)} samples @ {sample_rate}Hz ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f" Lỗi load audio: {e}")
            return jsonify({'success': False, 'error': f'Lỗi đọc file audio: {str(e)}'}), 400
        
        # 3. Phân tích audio đầu vào
        input_properties = analyze_audio_properties_fast(noisy_input, sample_rate)
        
        # 4. Xử lý qua pipeline 3-stage (SIMPLIFIED CALL)
        logger.info(" Bắt đầu xử lý qua pipeline...")
        start_time = datetime.now()
        
        results = pipeline.process_audio(
            noisy_input=noisy_input_original,
            noisy_input_original=noisy_input_original,
            environment=environment,
            # Remove: output_device=output_device,
            sample_rate=sample_rate
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f" Xử lý hoàn thành trong {processing_time:.3f}s")
        
        # 5. Lưu các file kết quả
        file_paths = {}
        
        # Lưu input có nhiễu
        input_path = UPLOAD_FOLDER / f"input_noisy_{temp_id}.wav"
        sf.write(str(input_path), noisy_input, sample_rate)
        file_paths['input_noisy'] = f"/static/audio/input_noisy_{temp_id}.wav"
        
        # Lưu sau Stage 1 (mic processed)
        mic_path = UPLOAD_FOLDER / f"mic_processed_{temp_id}.wav"
        sf.write(str(mic_path), results['mic_processed'], sample_rate)
        file_paths['mic_processed'] = f"/static/audio/mic_processed_{temp_id}.wav"
        
        # Lưu final output ( Stage 2)
        final_path = UPLOAD_FOLDER / f"final_output_{temp_id}.wav"
        sf.write(str(final_path), results['final_output'], sample_rate)
        file_paths['final_output'] = f"/static/audio/final_output_{temp_id}.wav"
        
        # 6. Tính toán frequency response cho charts
        freq_input, mag_input = compute_frequency_response_fast(noisy_input, sample_rate)
        freq_final, mag_final = compute_frequency_response_fast(results['final_output'], sample_rate)
        
        # 7. Chuẩn bị response data (SIMPLIFIED)
        response_data = {
            'success': True,
            'processing_id': temp_id,
            'environment': environment,
            # Remove: 'output_device': output_device,
            'file_paths': file_paths,
            
            # Metadata xử lý
            'processing_metadata': {
                'total_time': processing_time,
                'stage_times': results['processing_metadata']['stage_times'],
                'real_time_factor': results['processing_metadata']['real_time_factor'],
                'latency_ms': results['processing_metadata']['latency_ms'],
                'audio_duration': duration,
                'sample_rate': sample_rate
            },
            
            # Metrics chất lượng
            'quality_metrics': {
                'estimated_snr_improvement_db': results['quality_metrics']['estimated_snr_improvement_db'],
                'estimated_noise_reduction_db': results['quality_metrics'].get('estimated_noise_reduction_db', 0),
                'input_snr_db': results['quality_metrics'].get('input_snr_db', 0),
                'output_snr_db': results['quality_metrics'].get('output_snr_db', 0),
                'level_change_db': results['quality_metrics'].get('level_change_db', 0),
                'input_peak_db': results['quality_metrics'].get('input_peak_db', 0),
                'output_peak_db': results['quality_metrics'].get('output_peak_db', 0)
            },
            
            # Properties audio đầu vào
            'input_properties': input_properties,
            
            # Data cho charts
            'frequency_analysis': {
                'frequencies': freq_input[:len(freq_input)//4].tolist(),
                'input_spectrum': mag_input[:len(mag_input)//4].tolist(),
                'final_output_spectrum': mag_final[:len(mag_final)//4].tolist()
            },
            
            # Environment info only (remove device_info)
            'environment_info': ENVIRONMENT_CONFIGS[environment]
        }
        
        # Cleanup
        try:
            os.remove(str(temp_input_path))
        except:
            pass
        
        logger.info(f" API response ready. SNR improvement: {results['quality_metrics']['estimated_snr_improvement_db']:.1f}dB")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f" Lỗi xử lý audio: {e}")
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        }), 500


@app.route('/api/cleanup/<processing_id>')
def cleanup_files(processing_id):
    """Cleanup các file tạm thời"""
    try:
        file_patterns = [
            f"original_{processing_id}.wav",
            f"noisy_{processing_id}.wav", 
            f"processed_{processing_id}.wav",
            f"final_{processing_id}.wav"
        ]
        
        for pattern in file_patterns:
            file_path = UPLOAD_FOLDER / pattern
            if file_path.exists():
                os.remove(str(file_path))
        
        return jsonify({'success': True, 'message': 'Files cleaned up'})
    except Exception as e:
        logger.error(f" Lỗi cleanup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('base.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Tạo thư mục cần thiết
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Lấy port từ environment hoặc dùng mặc định
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    print("\n" + "="*60)
    print(" NOISE FILTERING - WEB INTERFACE")
    print("="*60)
    print(f"📡 Server: http://localhost:{port}")
    print(" Nhấn Ctrl+C để dừng")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=port)