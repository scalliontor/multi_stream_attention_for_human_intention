# ⚡ Speed Improvements

## 🚀 **Multiprocessing Preprocessing**

### **Before**: 15 hours (single CPU core)
### **After**: ~2-3 hours (all CPU cores)

## 🔧 **How to Use:**

### **Automatic (uses all available CPUs - 1)**:
```bash
python preprocess.py --frames_dir frames --annotations annotations.json --output processed_data
```

### **Manual (specify number of workers)**:
```bash
# Use 16 workers
python preprocess.py --frames_dir frames --annotations annotations.json --output processed_data --workers 16

# Use 32 workers (if you have many CPUs)
python preprocess.py --frames_dir frames --annotations annotations.json --output processed_data --workers 32
```

### **Test with small batch first**:
```bash
python preprocess.py --frames_dir frames --annotations annotations.json --output processed_data --max_videos 100 --workers 8
```

## 📊 **Expected Speed:**

| CPUs | Time for 220K videos |
|------|---------------------|
| 1    | ~15 hours          |
| 4    | ~4 hours           |
| 8    | ~2 hours           |
| 16   | ~1 hour            |
| 32   | ~30 minutes        |

## 💡 **Why MediaPipe Can't Use GPU:**

MediaPipe hand detection is **CPU-only** - it doesn't support GPU acceleration. However:
- ✅ **Multiprocessing** gives 4-16x speedup
- ✅ Each worker processes videos independently
- ✅ Scales linearly with CPU cores

## ⚠️ **Memory Considerations:**

- Each worker loads its own MediaPipe model (~200MB)
- 16 workers = ~3GB RAM
- 32 workers = ~6GB RAM
- Adjust `--workers` based on your RAM

## 🎯 **Recommended Settings:**

### **High-end server (32+ cores, 64GB+ RAM)**:
```bash
python preprocess.py --workers 32
```

### **Mid-range server (16 cores, 32GB RAM)**:
```bash
python preprocess.py --workers 16
```

### **Desktop (8 cores, 16GB RAM)**:
```bash
python preprocess.py --workers 8
```

### **Laptop (4 cores, 8GB RAM)**:
```bash
python preprocess.py --workers 4
```
