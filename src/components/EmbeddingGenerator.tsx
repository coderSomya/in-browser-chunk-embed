
import React, { useState, useRef } from 'react';
import { pipeline } from '@huggingface/transformers';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Upload, FileText, Brain, Download } from 'lucide-react';

interface Chunk {
  id: number;
  text: string;
  embedding?: number[];
}

const EmbeddingGenerator = () => {
  const [file, setFile] = useState<File | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [model, setModel] = useState<any>(null);
  const [embeddings, setEmbeddings] = useState<Chunk[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const chunkText = (text: string, chunkSize: number = 500): Chunk[] => {
    const words = text.split(' ');
    const chunks: Chunk[] = [];
    
    for (let i = 0; i < words.length; i += chunkSize) {
      const chunk = words.slice(i, i + chunkSize).join(' ');
      if (chunk.trim()) {
        chunks.push({
          id: chunks.length + 1,
          text: chunk.trim()
        });
      }
    }
    
    return chunks;
  };

  const readPDFFile = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async (e) => {
        try {
          const arrayBuffer = e.target?.result as ArrayBuffer;
          const uint8Array = new Uint8Array(arrayBuffer);
          
          // Simple PDF text extraction (basic implementation)
          const text = new TextDecoder().decode(uint8Array);
          
          // Extract readable text between common PDF markers
          const textContent = text
            .replace(/[\x00-\x1F\x7F-\x9F]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
          
          resolve(textContent);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  };

  const readTextFile = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setIsProcessing(true);
    setProgress(10);

    try {
      let text = '';
      
      if (uploadedFile.type === 'application/pdf') {
        text = await readPDFFile(uploadedFile);
      } else {
        text = await readTextFile(uploadedFile);
      }

      setProgress(30);
      
      const fileChunks = chunkText(text);
      setChunks(fileChunks);
      setProgress(50);
      
      console.log(`Generated ${fileChunks.length} chunks from file`);
    } catch (error) {
      console.error('Error processing file:', error);
      alert('Error processing file. Please try again.');
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const generateEmbeddings = async () => {
    if (chunks.length === 0) return;

    setIsProcessing(true);
    setProgress(0);

    try {
      // Load the embedding model (using wasm device for browser compatibility)
      console.log('Loading embedding model...');
      const extractor = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
        { device: 'wasm' }
      );
      
      setModel(extractor);
      setProgress(20);

      const chunksWithEmbeddings: Chunk[] = [];
      
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        console.log(`Processing chunk ${i + 1}/${chunks.length}`);
        
        // Generate embedding for this chunk
        const embedding = await extractor(chunk.text, { pooling: 'mean', normalize: true });
        
        chunksWithEmbeddings.push({
          ...chunk,
          embedding: embedding.data
        });
        
        setProgress(20 + (80 * (i + 1)) / chunks.length);
      }
      
      setEmbeddings(chunksWithEmbeddings);
      console.log('All embeddings generated successfully!');
      
    } catch (error) {
      console.error('Error generating embeddings:', error);
      alert('Error generating embeddings. Please try again.');
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  };

  const downloadEmbeddings = () => {
    if (embeddings.length === 0) return;
    
    const data = embeddings.map(chunk => ({
      id: chunk.id,
      text: chunk.text,
      embedding: chunk.embedding
    }));
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `embeddings_${file?.name || 'document'}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-white text-black p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4">Text Embedding Generator</h1>
          <p className="text-lg text-gray-600">Upload a document, chunk it, and generate embeddings in your browser</p>
        </div>

        <div className="grid gap-6">
          {/* File Upload */}
          <Card className="border-2 border-black">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                1. Upload File
              </CardTitle>
            </CardHeader>
            <CardContent>
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf"
                onChange={handleFileUpload}
                className="hidden"
              />
              <Button 
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                className="w-full border-black hover:bg-black hover:text-white"
                disabled={isProcessing}
              >
                {file ? `Selected: ${file.name}` : 'Choose PDF or Text File'}
              </Button>
              {file && (
                <div className="mt-4 p-4 bg-gray-50 border border-black">
                  <p><strong>File:</strong> {file.name}</p>
                  <p><strong>Size:</strong> {(file.size / 1024).toFixed(2)} KB</p>
                  <p><strong>Type:</strong> {file.type || 'text/plain'}</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Chunking Results */}
          {chunks.length > 0 && (
            <Card className="border-2 border-black">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="w-5 h-5" />
                  2. Text Chunks ({chunks.length} chunks)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-40 overflow-y-auto border border-black p-4">
                  {chunks.slice(0, 3).map((chunk) => (
                    <div key={chunk.id} className="mb-4 pb-4 border-b border-gray-300 last:border-b-0">
                      <p className="font-semibold">Chunk {chunk.id}:</p>
                      <p className="text-sm text-gray-700 mt-1">
                        {chunk.text.substring(0, 200)}...
                      </p>
                    </div>
                  ))}
                  {chunks.length > 3 && (
                    <p className="text-center text-gray-500 italic">
                      ... and {chunks.length - 3} more chunks
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Generate Embeddings */}
          {chunks.length > 0 && (
            <Card className="border-2 border-black">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5" />
                  3. Generate Embeddings
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Button
                  onClick={generateEmbeddings}
                  disabled={isProcessing || embeddings.length > 0}
                  className="w-full bg-black text-white hover:bg-gray-800"
                >
                  {isProcessing ? 'Generating Embeddings...' : 
                   embeddings.length > 0 ? 'Embeddings Generated ✓' : 
                   'Generate Embeddings'}
                </Button>
                {isProcessing && (
                  <div className="mt-4">
                    <Progress value={progress} className="w-full" />
                    <p className="text-center mt-2 text-sm">{Math.round(progress)}% complete</p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Results */}
          {embeddings.length > 0 && (
            <Card className="border-2 border-black">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Download className="w-5 h-5" />
                  4. Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-gray-50 border border-black">
                    <p><strong>Total Chunks:</strong> {embeddings.length}</p>
                    <p><strong>Embedding Dimension:</strong> {embeddings[0]?.embedding?.length || 'N/A'}</p>
                    <p><strong>Model Used:</strong> all-MiniLM-L6-v2</p>
                  </div>
                  
                  <Button
                    onClick={downloadEmbeddings}
                    className="w-full bg-black text-white hover:bg-gray-800"
                  >
                    Download Embeddings JSON
                  </Button>

                  <div className="max-h-60 overflow-y-auto border border-black p-4">
                    <h4 className="font-semibold mb-2">Sample Embeddings:</h4>
                    {embeddings.slice(0, 2).map((chunk) => (
                      <div key={chunk.id} className="mb-4 pb-4 border-b border-gray-300 last:border-b-0">
                        <p className="font-semibold">Chunk {chunk.id}:</p>
                        <p className="text-xs text-gray-600 mb-2">
                          "{chunk.text.substring(0, 100)}..."
                        </p>
                        <p className="text-xs font-mono text-gray-500">
                          Embedding: [{chunk.embedding?.slice(0, 5).map(n => n.toFixed(4)).join(', ')}...]
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default EmbeddingGenerator;
