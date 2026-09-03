# Owner: S2
# Module: INT8 Post-Training Quantization
# Input:  ml/models/naad_kws_float.h5
# Output: ml/models/naad_kws_int8.tflite
# Last validated: [date, commit]

import tensorflow as tf
import numpy as np

def quantize_model(float_model_path: str, calibration_data_path: str, output_path: str) -> float:
    """
    Full-integer INT8 quantization.

    CRITICAL: calibration data MUST be real MFCC tensors from your train set.
    Random tensors will give wrong scale estimates and destroy accuracy.
    
    Args:
        float_model_path: path to naad_kws_float.h5
        calibration_data_path: path to train_X.npy (real MFCC tensors)
        output_path: destination path for .tflite file
    Returns:
        size_kb: size of quantized model in KB
    """
    model = tf.keras.models.load_model(float_model_path)

    # Load real calibration data (min 200 samples)
    X_calib = np.load(calibration_data_path).astype(np.float32)
    print(f"Calibration samples: {len(X_calib)}")

    def representative_data_gen():
        indices = np.random.choice(len(X_calib), min(200, len(X_calib)), replace=False)
        for i in indices:
            yield [X_calib[i:i+1]]  # shape (1, 49, 13, 1), float32

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset    = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.int8    # ESP32 feeds int8
    converter.inference_output_type     = tf.int8    # ESP32 reads int8

    print("Converting to INT8...")
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"INT8 model saved: {output_path}")
    print(f"Model size: {size_kb:.1f} KB")
    return size_kb


def validate_int8_accuracy(float_path: str, tflite_path: str,
                            X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Compare float vs INT8 accuracy on test set."""
    # INT8 inference
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get quantization params
    in_scale     = input_details[0]['quantization'][0]
    in_zero_pt   = input_details[0]['quantization'][1]
    out_scale    = output_details[0]['quantization'][0]
    out_zero_pt  = output_details[0]['quantization'][1]

    correct = 0
    for i in range(len(X_test)):
        # Quantize input: float -> int8
        x = X_test[i:i+1].astype(np.float32)
        x_int8 = np.round(x / in_scale + in_zero_pt).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], x_int8)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize output
        proba = (output.astype(np.float32) - out_zero_pt) * out_scale
        pred  = np.argmax(proba)
        if pred == y_test[i]:
            correct += 1

    int8_accuracy = correct / len(y_test)
    print(f"INT8 model test accuracy: {int8_accuracy:.4f} ({int8_accuracy*100:.1f}%)")
    return int8_accuracy


if __name__ == "__main__":
    size = quantize_model(
        float_model_path      = "ml/models/naad_kws_float.h5",
        calibration_data_path = "ml/data/train_X.npy",
        output_path           = "ml/models/naad_kws_int8.tflite"
    )

    X_test = np.load("ml/data/test_X.npy")
    y_test = np.load("ml/data/test_y.npy")
    validate_int8_accuracy(
        "ml/models/naad_kws_float.h5",
        "ml/models/naad_kws_int8.tflite",
        X_test, y_test
    )
