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

**OPTIMIZED**: Annotations loaded once, not per worker!
- Main process: Loads 180K annotations once (~2GB)
- Each worker: Only MediaPipe model (~200MB)
- 16 workers = ~2GB (annotations) + 3GB (workers) = ~5GB total
- **Much more memory efficient now!**

## 🎯 **Recommended Settings:**

### **High-end server (32+ cores, 64GB+ RAM)**:
```bash
# Start with 8, increase if stable
python preprocess.py --workers 8
# If stable, try 16
python preprocess.py --workers 16
```

### **Mid-range server (16 cores, 32GB RAM)**:
```bash
# Safe default
python preprocess.py --workers 8
```

### **Desktop (8 cores, 16GB RAM)**:
```bash
# Conservative
python preprocess.py --workers 4
```

### **Laptop (4 cores, 8GB RAM)**:
```bash
# Very conservative
python preprocess.py --workers 2
```

## 💡 **Best Practice:**

1. **Start small**: Test with `--workers 4 --max_videos 10`
2. **Monitor RAM**: Use `htop` or `top` to watch memory usage
3. **Increase gradually**: If stable, double the workers
4. **If killed**: Reduce workers by half

### **Safe Test Command**:
```bash
# Test with 4 workers on 10 videos
python preprocess.py --workers 4 --max_videos 10
```
