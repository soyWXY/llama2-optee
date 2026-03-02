import sys
import struct
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def encrypt_file_fixed_iv(input_path, output_path, num_thread=2):
    key = b'\x00' * 32
    fixed_iv = b'\x00' * 12 
    
    aesgcm = AESGCM(key)
    
    with open(input_path, "rb") as f:
        plaintext = f.read()
        
    total_size = len(plaintext)
    nblock = num_thread
    
    # 計算基礎區塊大小 (向上取整)
    chunk_size = (total_size + nblock - 1) // nblock
    
    encrypted_blocks = []
    intervals = [0]
    
    # 執行負載離散與獨立加密
    for i in range(nblock):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_size)
        chunk_pt = plaintext[start_idx:end_idx]
            
        # AESGCM 預設輸出格式: Ciphertext + 16 Bytes Tag
        chunk_enc = aesgcm.encrypt(fixed_iv, chunk_pt, None)
        encrypted_blocks.append(chunk_enc)
        intervals.append(intervals[-1] + len(chunk_enc))
        
    header_size = 4 + (nblock + 1) * 4
    
    # 執行二進位序列化
    with open(output_path, "wb") as f:
        total_size
        # 1. 寫入 total_size (uint32_t, Little-Endian)
        f.write(struct.pack('<I', total_size))

        # 2. 寫入 nblock (uint32_t, Little-Endian)
        f.write(struct.pack('<I', nblock))
        
        # 3. 寫入 interval array (uint32_t, Little-Endian)
        for offset in intervals:
            f.write(struct.pack('<I', offset))
            
        # 4. 寫入 Payload 區段 (不含 Padding 直接串接)
        for enc_chunk in encrypted_blocks:
            f.write(enc_chunk)
            
    print("=========================================")
    print("Encryption Process Completed (Parallel Segment Format)")
    print(f"Input File     : {total_size} Bytes")
    print(f"Threads/Blocks : {nblock}")
    print(f"Max Chunk Size : {chunk_size} Bytes (Plaintext)")
    print(f"Intervals      : {intervals}")
    print("=========================================")

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    thread_count = int(sys.argv[3]) if len(sys.argv) == 4 else 2
    
    if thread_count <= 0:
        print("Error: num_thread must be greater than 0.")
        sys.exit(1)
        
    encrypt_file_fixed_iv(input_file, output_file, num_thread=thread_count)