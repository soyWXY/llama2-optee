import sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def encrypt_file_fixed_iv(input_path, output_path):
    # 定義 32 Bytes 靜態金鑰與 12 Bytes 靜態 IV
    key = b'\x00' * 32
    fixed_iv = b'\x00' * 12 
    
    aesgcm = AESGCM(key)
    
    with open(input_path, "rb") as f:
        plaintext = f.read()
        
    encrypted_data = aesgcm.encrypt(fixed_iv, plaintext, None)
    
    # 僅寫入 Tag (末 16 Bytes) 與 Ciphertext
    tag = encrypted_data[-16:]
    ciphertext = encrypted_data[:-16]
    
    with open(output_path, "wb") as f:
        f.write(ciphertext)
        f.write(tag)
        
    print(f"Encryption done. Tag(16B) + CT({len(ciphertext)}B) written to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    encrypt_file_fixed_iv(sys.argv[1], sys.argv[2])