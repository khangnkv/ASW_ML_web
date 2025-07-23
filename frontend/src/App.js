import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Alert, Spinner, Badge, Dropdown, ButtonGroup, Modal, ProgressBar } from 'react-bootstrap';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { saveAs } from 'file-saver';
import { FiUpload, FiDownload, FiDatabase, FiSun, FiMoon, FiClock, FiInfo, FiGrid, FiCode } from 'react-icons/fi';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Import components
import ProjectFilters from './components/ProjectFilters';
import DataStats from './components/DataStats';

function App() {
  const [fileData, setFileData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    // Persist mode in localStorage
    return localStorage.getItem('darkMode') === 'true';
  });
  const [showPreview, setShowPreview] = useState(true);
  const [fileInfo, setFileInfo] = useState(null);
  const [storageStats, setStorageStats] = useState(null);
  
  // Export state
  const [exportLoading, setExportLoading] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [exportFormat, setExportFormat] = useState('');
  
  // Project filtering state
  const [selectedProject, setSelectedProject] = useState(null);
  const [availableProjects, setAvailableProjects] = useState([]);
  const [fullDataset, setFullDataset] = useState(null);
  const [filteredData, setFilteredData] = useState(null);
  const [filterLoading, setFilterLoading] = useState(false);
  const [previewNumRows, setPreviewNumRows] = useState(5);
  const [previewSection, setPreviewSection] = useState('both'); // 'head', 'tail', 'both'

  // Prediction results state
  const [predictionResultsInfo, setPredictionResultsInfo] = useState(null);

  // Add new state variables for explainability
  const [explainabilityData, setExplainabilityData] = useState(null);
  const [explainabilityLoading, setExplainabilityLoading] = useState(false);
  const [showExplainability, setShowExplainability] = useState(false);
  const [explainableProjects, setExplainableProjects] = useState([]);

  useEffect(() => {
    // Fetch available models on component mount
    fetchAvailableModels();
    fetchStorageStats();
  }, []);

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
    document.body.classList.toggle('light-mode', !darkMode);
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get('/api/models');
      // We'll use this data later if needed
      console.log('Available models:', response.data.available_models);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const fetchStorageStats = async () => {
    try {
      const response = await axios.get('/api/storage/stats');
      setStorageStats(response.data);
    } catch (error) {
      console.error('Error fetching storage stats:', error);
    }
  };

  // Extract unique projects from data
  const extractProjects = (data) => {
    if (!data || data.length === 0) return [];
    
    const projects = new Set();
    data.forEach(row => {
      if (row.projectid) {
        projects.add(row.projectid.toString());
      }
    });
    
    return Array.from(projects);
  };

  // Filter data by project
  const filterDataByProject = (data, projectId) => {
    if (!data || !projectId) return data;
    
    setFilterLoading(true);
    
    // Use setTimeout to prevent UI blocking for large datasets
    setTimeout(() => {
      const filtered = data.filter(row => 
        row.projectid && row.projectid.toString() === projectId.toString()
      );
      setFilteredData(filtered);
      setFilterLoading(false);
    }, 0);
  };

  // Handle project selection
  const handleProjectFilter = (projectId) => {
    setSelectedProject(projectId);
    if (projectId) {
      filterDataByProject(fullDataset, projectId);
    } else {
      setFilteredData(fullDataset);
    }
  };

  // Handle filter reset
  const handleFilterReset = () => {
    setSelectedProject(null);
    setFilteredData(fullDataset);
  };

  // Current data (either filtered or full dataset)
  const currentData = selectedProject ? filteredData : fullDataset;

  // --- Preview Controls ---
  const previewOptions = [5, 10, 50];
  const sectionOptions = [
    { value: 'head', label: 'Head' },
    { value: 'tail', label: 'Tail' },
    { value: 'both', label: 'Head & Tail' },
  ];

  // Preview rows logic
  const getPreviewRows = () => {
    if (!currentData || currentData.length === 0) return [];
    const n = previewNumRows;
    if (previewSection === 'head') {
      return currentData.slice(0, n);
    } else if (previewSection === 'tail') {
      return currentData.slice(-n);
    } else { // both
      if (currentData.length <= 2 * n) {
        return currentData;
      }
      const firstN = currentData.slice(0, n);
      const lastN = currentData.slice(-n);
      return [...firstN, ...lastN];
    }
  };
  
  const previewRows = getPreviewRows();

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      setShowPreview(true);
      setSelectedProject(null);

      // Make the upload request
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      console.log('Uploading file to:', `${API_URL}/api/upload_uuid`);
      
      const response = await axios.post(
        `${API_URL}/api/upload_uuid`,
        formData,
        { 
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 60000 // 60 second timeout
        }
      );

      console.log("UPLOAD RESPONSE:", response.data);

      // Check if response has the expected structure
      if (!response.data) {
        console.error('Empty response from server');
        setError('Upload failed: Empty response from server.');
        return;
      }

      // Defensive: check if response contains expected fields
      const { filename, preview, full_dataset, file_info, success } = response.data;
      
      if (!success) {
        console.error('Upload not successful:', response.data);
        setError(response.data.error || 'Upload failed: Server returned unsuccessful response.');
        return;
      }
      
      if (!preview || !full_dataset) {
        console.error('Missing required data in response:', {
          hasPreview: !!preview,
          hasFullDataset: !!full_dataset,
          hasFileInfo: !!file_info,
          responseKeys: Object.keys(response.data)
        });
        setError('Upload failed: Missing preview or dataset in server response.');
        return;
      }

      // Extract projects from full_dataset
      const projects = extractProjects(full_dataset);

      setFileData(full_dataset);
      setFullDataset(full_dataset);
      setAvailableProjects(projects);
      setFileInfo(file_info);

      console.log('Upload successful:', {
        filename,
        previewRows: preview.length,
        fullDatasetRows: full_dataset.length,
        originalName: file.name,
        fileInfo: file_info,
        availableProjects: projects,
    });
    
    } catch (error) {
      console.error('Upload error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        headers: error.response?.headers
      });
      
      if (error.code === 'ECONNABORTED') {
        setError('Upload timeout: File too large or connection slow. Please try again.');
      } else if (error.response?.data?.error) {
        setError(`Upload failed: ${error.response.data.error}`);
      } else if (error.response?.status) {
        setError(`Upload failed: Server error (${error.response.status})`);
      } else {
        setError(`Upload failed: ${error.message || 'Unknown error'}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    multiple: false,
  });

  // Function to fetch all prediction result files
  const fetchPredictionResultsFiles = async () => {
    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await axios.get(`${API_URL}/api/prediction_results`);
      
      console.log('Available prediction result files:', response.data);
      return response.data.prediction_results || [];
    } catch (error) {
      console.error('Error fetching prediction result files:', error);
      return [];
    }
  };

  // Function to download a specific prediction result file
  const downloadPredictionResults = async (filename) => {
    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      
      // Try different endpoints for downloading
      let response;
      try {
        // First try the export endpoint
        response = await axios.get(`${API_URL}/api/export/csv/prediction_results/${filename}`, {
          responseType: 'blob'
        });
      } catch (exportError) {
        // Fallback to direct file download
        response = await axios.get(`${API_URL}/api/prediction_results/${filename}/download`, {
          responseType: 'blob'
        });
      }
      
      saveAs(response.data, filename);
      console.log('Downloaded prediction results:', filename);
    } catch (error) {
      console.error('Error downloading prediction results:', error);
      setError(`Failed to download ${filename}: ${error.message}`);
    }
  };

  // NEW: Function to load prediction results from saved file
  const loadPredictionResults = async (filename) => {
    try {
      console.log('Loading prediction results from saved file:', filename);
      
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await axios.get(`${API_URL}/api/prediction_results/${filename}`);
      
      console.log('Loaded prediction results response:', response.data);
      
      if (response.data.success && response.data.complete_dataset) {
        // Update the full dataset with data from the saved file
        setFullDataset(response.data.complete_dataset);
        
        // Reset project filter to show all data
        setSelectedProject(null);
        setFilteredData(null);
        
        // Extract available projects from the loaded data
        if (response.data.available_projects) {
          setAvailableProjects(response.data.available_projects);
        } else {
          // Fallback: extract projects from dataset
          const projects = extractProjects(response.data.complete_dataset);
          setAvailableProjects(projects);
        }
        
        // Hide data preview, only show prediction results
        setShowPreview(false);
        
        console.log(`Loaded prediction results successfully: ${response.data.stats.total_rows} rows, ${response.data.stats.unique_projects} projects`);
      } else {
        setError('Failed to load prediction results from saved file');
        console.error('Invalid response structure:', response.data);
      }
      
    } catch (error) {
      console.error('Error loading prediction results:', error);
      setError(`Failed to load prediction results: ${error.message}`);
    }
  };

  const generatePredictions = async () => {
    if (!fileInfo?.filename) {
      setError('No file uploaded for prediction');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      console.log('Generating predictions for file:', fileInfo.filename);
      
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await axios.post(
        `${API_URL}/api/predict_workflow`,
        { filename: fileInfo.filename },
        { headers: { 'Content-Type': 'application/json' } }
      );

      console.log('Prediction response:', response.data);

      if (response.data.predictions) {
        setPredictions(response.data.predictions);
        
        // Store prediction results info
        if (response.data.prediction_results_filename) {
          setPredictionResultsInfo({
            filename: response.data.prediction_results_filename,
            path: response.data.prediction_results_path,
            created_at: new Date().toISOString()
          });
          
          // Load the actual prediction results data from the saved file
          await loadPredictionResults(response.data.prediction_results_filename);
        }
        
        // Update fileInfo to include processed filename
        if (response.data.processed_filename) {
          setFileInfo(prev => ({
            ...prev,
            processed_filename: response.data.processed_filename,
            prediction_results_filename: response.data.prediction_results_filename
          }));
        }
        
        // Update the full dataset with the complete prediction results
        if (response.data.complete_dataset) {
          setFullDataset(response.data.complete_dataset);
          // Reset project filter to show all data
          setSelectedProject(null);
          setFilteredData(null);
          // Extract available projects from the new data
          const projects = extractProjects(response.data.complete_dataset);
          setAvailableProjects(projects);
          
          // Hide data preview, only show prediction results
          setShowPreview(false);
        }
        
        console.log(`Generated ${response.data.predictions.length} predictions`);
        console.log('Prediction results saved to:', response.data.prediction_results_path);
      } else {
        setError('No predictions returned from server');
      }

    } catch (error) {
      console.error('Prediction error:', error);
      setError(error.response?.data?.error || error.message || 'Error generating predictions');
    } finally {
      setLoading(false);
    }
  };

  // --- Export logic with progress tracking ---
  const exportResults = async (format) => {
    if (!fileInfo?.filename) {
      setError('No file available for export');
      return;
    }

    setExportLoading(true);
    setExportProgress(0);
    setExportFormat(format.toUpperCase());
    setError(null);

    try {
      // Simulate progress updates during export
      const progressInterval = setInterval(() => {
        setExportProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const startTime = Date.now();
      
      // Make the export request
      const response = await axios.get(`/api/export/${format}/${fileInfo.filename}`, {
        responseType: 'blob',
        onDownloadProgress: (progressEvent) => {
          if (progressEvent.lengthComputable) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setExportProgress(Math.max(progress, 90)); // Ensure we show at least 90%
          }
        }
      });

      // Clear the interval and set progress to 100%
      clearInterval(progressInterval);
      setExportProgress(100);

      // Determine file extension and create filename
      let extension = format;
      let mimeType = response.headers['content-type'] || '';
      
      if (format === 'xlsx') {
        extension = 'xlsx';
      } else if (format === 'csv') {
        extension = 'csv';
      } else if (format === 'json') {
        extension = 'json';
      }

      // Create a more descriptive filename
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:\-T]/g, '');
      const originalName = fileInfo.originalName ? 
        fileInfo.originalName.split('.')[0] : 
        fileInfo.filename.split('.')[0];
      const filename = `predictions_${originalName}_${timestamp}.${extension}`;

      // Handle different response types
      let blob;
      if (format === 'json' && mimeType.includes('application/json')) {
        // For JSON, we might get a JSON response that we need to convert to blob
        const jsonData = await response.data.text();
        blob = new Blob([jsonData], { type: 'application/json' });
      } else {
        blob = response.data;
      }

      // Trigger download
      saveAs(blob, filename);

      // Calculate and show completion time
      const completionTime = ((Date.now() - startTime) / 1000).toFixed(1);
      
      // Show success message briefly
      setTimeout(() => {
        setExportLoading(false);
        setExportProgress(0);
        setExportFormat('');
      }, 1000);

      // Show a success toast or message
      console.log(`Export completed in ${completionTime}s: ${filename}`);
      
    } catch (error) {
      console.error('Export error:', error);
      setExportLoading(false);
      setExportProgress(0);
      setExportFormat('');
      
      if (error.response?.data) {
        try {
          const errorText = await error.response.data.text();
          const errorObj = JSON.parse(errorText);
          setError(`Export failed: ${errorObj.error || 'Unknown error'}`);
        } catch {
          setError('Export failed: Server error');
        }
      } else {
        setError(`Export failed: ${error.message || 'Network error'}`);
      }
    }
  };

  // Add function to fetch explainable projects
  const fetchExplainableProjects = async () => {
    if (!fileInfo?.prediction_results_filename) return;
    
    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await axios.get(`${API_URL}/api/explainability/${fileInfo.prediction_results_filename}/projects`);
      
      setExplainableProjects(response.data.available_projects || []);
    } catch (error) {
      console.error('Error fetching explainable projects:', error);
    }
  };

  // Add function to get explainability analysis
  const getExplainabilityAnalysis = async (projectId) => {
    if (!fileInfo?.prediction_results_filename) return;
    
    setExplainabilityLoading(true);
    setError(null);
    
    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await axios.get(`${API_URL}/api/explainability/${fileInfo.prediction_results_filename}/${projectId}`);
      
      setExplainabilityData(response.data);
      setShowExplainability(true);
      console.log('Explainability analysis:', response.data);
    } catch (error) {
      console.error('Error getting explainability analysis:', error);
      setError(`Failed to load explainability analysis: ${error.response?.data?.error || error.message}`);
    } finally {
      setExplainabilityLoading(false);
    }
  };

  // Add useEffect to fetch explainable projects when predictions are ready
  useEffect(() => {
    if (predictions && fileInfo?.prediction_results_filename) {
      fetchExplainableProjects();
    }
  }, [predictions, fileInfo]);

  const cleanup = async () => {
    setFileData(null);
    setPredictions(null);
    setFileInfo(null);
    setError(null);
    setShowPreview(true);
    
    // Reset filter state
    setSelectedProject(null);
    setAvailableProjects([]);
    setFullDataset(null);
    setFilteredData(null);
    setFilterLoading(false);
    
    // Reset export state
    setExportLoading(false);
    setExportProgress(0);
    setExportFormat('');
    
    // Reset prediction results state
    setPredictionResultsInfo(null);
    
    // Reset explainability state
    setExplainabilityData(null);
    setExplainabilityLoading(false);
    setShowExplainability(false);
    setExplainableProjects([]);
  };

  const renderPreviewTable = (data, title) => {
    if (!data || data.length === 0) return null;
    
    let columns = Object.keys(data[0]);
    // Move prediction columns to the end
    const predictionCols = ['has_booked_prediction', 'prediction_confidence'];
    const otherCols = columns.filter(c => !predictionCols.includes(c));
    const existingPredCols = predictionCols.filter(c => columns.includes(c));
    columns = [...otherCols, ...existingPredCols];
    
    // Helper function to format cell values
    const formatCellValue = (value, column) => {
      if (value === null || value === undefined || value === '') return '';
      
      if (column === 'has_booked_prediction') {
        if (typeof value === 'number') {
          const prediction = value >= 0.5 ? 'Potential customer' : 'Not potential customer';
          return prediction;
        }
      }
      
      if (column === 'prediction_confidence') {
        if (typeof value === 'number') {
          return (value * 100).toFixed(1) + '%';
        }
      }
      
      if (typeof value === 'boolean') {
        return value ? 'Yes' : 'No';
      }
      
      return value.toString();
    };
    
    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            {title}
            {filterLoading && (
              <Spinner animation="border" size="sm" className="ms-2" />
            )}
          </h5>
        </Card.Header>
        <Card.Body className="p-0">
          <div className="table-responsive">
            <table className="table table-striped table-hover mb-0">
              <thead className="table-primary">
                <tr>
                  {columns.map((column) => (
                    <th key={column} style={{
                      backgroundColor: predictionCols.includes(column) ? '#d4edda' : '',
                      fontWeight: predictionCols.includes(column) ? 'bold' : 'normal',
                      fontSize: predictionCols.includes(column) ? '1.05em' : 'inherit',
                      color: predictionCols.includes(column) ? '#1e7e34' : ''
                    }}>
                      {column === 'has_booked_prediction' ? 'Prediction' : 
                       column === 'prediction_confidence' ? 'Confidence' : column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, index) => (
                  <tr key={index}>
                    {columns.map((column) => (
                      <td key={column} style={{
                        backgroundColor: predictionCols.includes(column) ? '#f8f9fa' : '',
                        fontWeight: predictionCols.includes(column) ? 'bold' : 'normal',
                        fontSize: predictionCols.includes(column) ? '1.05em' : 'inherit',
                        color: (column === 'has_booked_prediction' || column === 'prediction_confidence') && 
                               row['has_booked_prediction'] >= 0.5 ? '#28a745' : 
                               (column === 'has_booked_prediction' || column === 'prediction_confidence') && 
                               row['has_booked_prediction'] < 0.5 ? '#dc3545' : ''
                      }}>
                        {formatCellValue(row[column], column)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card.Body>
      </Card>
    );
  };

  // --- Preview controls UI ---
  const renderPreviewControls = () => (
    <div className="d-flex align-items-center gap-3 mb-2">
      <span>Preview:</span>
      <Dropdown as={ButtonGroup} onSelect={val => setPreviewNumRows(Number(val))}>
        <Button variant="outline-primary">{previewNumRows} rows</Button>
        <Dropdown.Toggle split variant="outline-primary" id="dropdown-split-basic" />
        <Dropdown.Menu>
          {previewOptions.map(opt => (
            <Dropdown.Item key={opt} eventKey={opt}>{opt} rows</Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown>
      <Dropdown as={ButtonGroup} onSelect={val => setPreviewSection(val)}>
        <Button variant="outline-secondary">{sectionOptions.find(o => o.value === previewSection)?.label}</Button>
        <Dropdown.Toggle split variant="outline-secondary" id="dropdown-split-section" />
        <Dropdown.Menu>
          {sectionOptions.map(opt => (
            <Dropdown.Item key={opt.value} eventKey={opt.value}>{opt.label}</Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown>
    </div>
  );

  const renderFileRetentionInfo = () => {
    if (!fileInfo) return null;

    const uploadDate = new Date(fileInfo.upload_timestamp);
    const deletionDate = new Date(fileInfo.deletion_date);
    const daysLeft = Math.ceil((deletionDate - new Date()) / (1000 * 60 * 60 * 24));

    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <FiClock className="me-2" />
            File Retention Information
          </h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={6}>
              <p><strong>Upload Time:</strong> {uploadDate.toLocaleString()}</p>
              <p><strong>Deletion Date:</strong> {deletionDate.toLocaleString()}</p>
            </Col>
            <Col md={6}>
              <p><strong>Days Remaining:</strong> 
                <Badge 
                  bg={daysLeft > 30 ? 'success' : daysLeft > 7 ? 'warning' : 'danger'}
                  className="ms-2"
                >
                  {daysLeft} days
                </Badge>
              </p>
              <p><strong>Status:</strong> 
                <Badge bg="info" className="ms-2">{fileInfo.status}</Badge>
              </p>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    );
  };

  const renderStorageStats = () => {
    if (!storageStats) return null;

    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <FiInfo className="me-2" />
            Storage Statistics
          </h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-primary">{storageStats.total_files}</div>
                <div className="text-muted">Total Files</div>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-info">{storageStats.total_size_mb} MB</div>
                <div className="text-muted">Total Size</div>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-success">{storageStats.retention_days}</div>
                <div className="text-muted">Retention Days</div>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center">
                <Button 
                  variant="outline-secondary" 
                  size="sm"
                  onClick={fetchStorageStats}
                >
                  <FiInfo className="me-1" />
                  Refresh
                </Button>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    );
  };

  // --- Compute and Render Prediction Statistics (per project or overall) ---
  const renderPredictionStats = () => {
    // Use filtered data (currentData) for per-project stats
    const data = currentData;
    if (!data || data.length === 0) return null;
    const total = data.length;
    const totalCols = data[0] ? Object.keys(data[0]).length : 0;
    // Count numeric and categorical columns
    let numericCols = 0, categoricalCols = 0;
    if (data[0]) {
      Object.keys(data[0]).forEach(col => {
        const val = data[0][col];
        if (typeof val === 'number' && !isNaN(val)) numericCols++;
        else categoricalCols++;
      });
    }
    const predCol = data.map(row => row.has_booked_prediction);
    const missingCount = predCol.filter(v => v === null || v === undefined || v === '' || (typeof v === 'number' && isNaN(v))).length;
    // Count each class - predictions are probabilities, so >=0.5 means likely to book
    const classCounts = { '0': 0, '1': 0 };
    predCol.forEach(v => {
      if (v !== null && v !== undefined && v !== '' && !isNaN(v)) {
        if (parseFloat(v) >= 0.5) {
          classCounts['1']++;
        } else {
          classCounts['0']++;
        }
      }
    });
    // Percent for each class
    const classPercents = {
      '0': total ? ((100 * classCounts['0']) / total).toFixed(2) + '%' : '0.00%',
      '1': total ? ((100 * classCounts['1']) / total).toFixed(2) + '%' : '0.00%'
    };
    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0" style={{ color: 'var(--bs-body-color)' }}>Project {selectedProject ? selectedProject : 'All'} Prediction Statistics</h5>
        </Card.Header>
        <Card.Body>
          <div style={{ fontSize: '1.1rem', color: 'var(--bs.body-color)' }}>
            <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', marginBottom: 8 }}>
              <div><span style={{ color: '#1976d2', fontWeight: 700, fontSize: 22 }}>{total}</span><br />Total Rows</div>
              <div><span style={{ color: '#00bcd4', fontWeight: 700, fontSize: 22 }}>{totalCols}</span><br />Total Columns</div>
              <div><span style={{ color: '#43a047', fontWeight: 700, fontSize: 22 }}>{numericCols}</span><br />Numeric Columns</div>
              <div><span style={{ color: '#ffc107', fontWeight: 700, fontSize: 22 }}>{categoricalCols}</span><br />Categorical Columns</div>
            </div>
            <div style={{ margin: '12px 0' }}>
              <span style={{ fontWeight: 600, background: 'var(--bs-secondary-bg, #f5f5f5)', borderRadius: 16, padding: '6px 16px', display: 'inline-block', color: 'var(--bs.body-color)' }}>
                {missingCount.toLocaleString()} missing values detected
              </span>
            </div>
            <div style={{ marginTop: 12, fontWeight: 600 }}>Prediction Results:</div>
            <div style={{ marginLeft: 12, marginTop: 4 }}>
              <div style={{ color: '#388e3c', marginBottom: 2 }}>
                Potential customers: <span style={{ fontWeight: 700 }}>{classCounts['1']}</span> <span style={{ color: '#888', fontWeight: 400 }}>({classPercents['1']})</span>
              </div>
              <div style={{ color: '#d32f2f' }}>
                Not potential customers: <span style={{ fontWeight: 700 }}>{classCounts['0']}</span> <span style={{ color: '#888', fontWeight: 400 }}>({classPercents['0']})</span>
              </div>
            </div>
          </div>
        </Card.Body>
      </Card>
    );
  };

  // --- Export Progress Modal ---
  const renderExportModal = () => (
    <Modal show={exportLoading} backdrop="static" keyboard={false} centered>
      <Modal.Header>
        <Modal.Title>
          <FiDownload className="me-2" />
          Exporting {exportFormat} File
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <div className="text-center">
          <p className="mb-3">Preparing your {exportFormat} file for download...</p>
          <ProgressBar 
            now={exportProgress} 
            label={`${exportProgress}%`}
            animated
            striped
            variant={exportProgress === 100 ? "success" : "primary"}
            style={{ height: '25px' }}
          />
          <div className="mt-3">
            <small className="text-muted">
              {exportProgress < 30 && "Processing data..."}
              {exportProgress >= 30 && exportProgress < 60 && "Formatting file..."}
              {exportProgress >= 60 && exportProgress < 90 && "Preparing download..."}
              {exportProgress >= 90 && exportProgress < 100 && "Almost ready..."}
              {exportProgress === 100 && "Complete! Download should start automatically."}
            </small>
          </div>
        </div>
      </Modal.Body>
    </Modal>
  );

  // Add explainability rendering function
  const renderExplainabilitySection = () => {
    if (!predictions || explainableProjects.length === 0) return null;

    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <FiInfo className="me-2" />
            Customer Analysis & Ideal Customer Insights
          </h5>
        </Card.Header>
        <Card.Body>
          <p className="text-muted mb-3">
            Analyze customer patterns and find ideal customers for each class (Potential vs Non-Potential)
          </p>
          
          <div className="d-flex flex-wrap gap-2 mb-3">
            {explainableProjects.map(project => (
              <Button
                key={project.project_id}
                variant="outline-info"
                size="sm"
                onClick={() => getExplainabilityAnalysis(project.project_id)}
                disabled={explainabilityLoading}
                className="d-flex align-items-center"
              >
                {explainabilityLoading ? (
                  <Spinner animation="border" size="sm" className="me-2" />
                ) : (
                  <FiGrid className="me-2" />
                )}
                Project {project.project_id}
                <Badge bg="success" className="ms-2">
                  {project.success_cases} success
                </Badge>
              </Button>
            ))}
          </div>
          
          {showExplainability && explainabilityData && (
            <Card className="mt-3">
              <Card.Header className="d-flex justify-content-between align-items-center">
                <h6 className="mb-0">
                  Project {explainabilityData.project_id} Analysis
                </h6>
                <Button
                  variant="outline-secondary"
                  size="sm"
                  onClick={() => setShowExplainability(false)}
                >
                  Close
                </Button>
              </Card.Header>
              <Card.Body>
                {/* Class Distribution Summary */}
                <Row className="mb-4">
                  <Col md={6}>
                    <div className="text-center">
                      <div className="h4 text-primary">{explainabilityData.total_samples}</div>
                      <small className="text-muted">Total Samples</small>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="text-center">
                      <div className="h4 text-success">{explainabilityData.class_distribution.class_1.count}</div>
                      <small className="text-muted">Potential Customers ({explainabilityData.class_distribution.class_1.percentage}%)</small>
                    </div>
                  </Col>
                  <Col md={3}>
                    <div className="text-center">
                      <div className="h4 text-danger">{explainabilityData.class_distribution.class_0.count}</div>
                      <small className="text-muted">Non-Potential ({explainabilityData.class_distribution.class_0.percentage}%)</small>
                    </div>
                  </Col>
                </Row>

                {/* Analysis for Both Classes - Show ALL Features */}
                {['class_1', 'class_0'].map(classKey => (
                  <div key={classKey} className="mb-5">
                    <h6 className={`text-${classKey === 'class_1' ? 'success' : 'danger'} mb-3`}>
                      {explainabilityData.class_distribution[classKey].label} - Complete Features Analysis
                    </h6>
                    
                    {explainabilityData.feature_analysis[classKey] && Object.keys(explainabilityData.feature_analysis[classKey]).length > 0 ? (
                      <>
                        {/* Show ALL Features for this class */}
                        <div className="row">
                          {Object.entries(explainabilityData.feature_analysis[classKey]).map(([feature, data]) => (
                            <div key={feature} className="col-md-6 mb-4">
                              <Card className="h-100">
                                <Card.Header className={`bg-${classKey === 'class_1' ? 'success' : 'danger'} text-white`}>
                                  <h6 className="mb-0">{feature}</h6>
                                  <small>({data.sample_size} samples, {data.total_unique} unique values)</small>
                                </Card.Header>
                                <Card.Body>
                                  <div className="mb-2">
                                    <strong>Top 5 Most Frequent Values:</strong>
                                  </div>
                                  {data.top_values.map((item, idx) => (
                                    <div 
                                      key={idx} 
                                      className={`p-2 mb-2 rounded ${item.is_top_frequent ? 'bg-warning bg-opacity-25 border border-warning' : 'bg-light'}`}
                                    >
                                      <div className="d-flex justify-content-between align-items-center">
                                        <div>
                                          <strong className={item.is_top_frequent ? 'text-warning-emphasis' : ''}>
                                            #{idx + 1}: {item.value}
                                            {item.is_top_frequent && <span className="ms-2 badge bg-warning text-dark">Most Frequent</span>}
                                          </strong>
                                        </div>
                                        <div className="text-end">
                                          <div className="fw-bold">{item.count} times</div>
                                          <div className="text-muted small">{item.percentage}%</div>
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </Card.Body>
                              </Card>
                            </div>
                          ))}
                        </div>
                        
                        {/* Ideal Customer for class_1 only */}
                        {classKey === 'class_1' && explainabilityData.ideal_customers && explainabilityData.ideal_customers.class_1 && (
                          <>
                            <h6 className="text-success mb-3 mt-4">
                              <i className="fas fa-crown me-2"></i>
                              Most Ideal Potential Customer Profile
                            </h6>
                            <Card className="border-success">
                              <Card.Body>
                                <Row className="mb-3">
                                  <Col md={3}>
                                    <strong>Customer ID:</strong> {explainabilityData.ideal_customers.class_1.customer_id}
                                  </Col>
                                  <Col md={3}>
                                    <strong>Match Score:</strong>{' '}
                                    <Badge bg="success">
                                      {explainabilityData.ideal_customers.class_1.match_percentage}%
                                    </Badge>
                                  </Col>
                                  <Col md={3}>
                                    <strong>Prediction:</strong>{' '}
                                    {(explainabilityData.ideal_customers.class_1.prediction_value * 100).toFixed(1)}%
                                  </Col>
                                  <Col md={3}>
                                    <strong>Features Matched:</strong>{' '}
                                    {explainabilityData.ideal_customers.class_1.matched_features.length}/
                                    {explainabilityData.ideal_customers.class_1.total_features_checked}
                                  </Col>
                                </Row>
                                
                                <h6 className="text-info mb-2">Matching Features:</h6>
                                <Row>
                                  {explainabilityData.ideal_customers.class_1.matched_features.map((feature, idx) => (
                                    <Col md={4} key={idx} className="mb-2">
                                      <div className={`p-2 rounded ${feature.frequency_rank === 1 ? 'bg-warning bg-opacity-25 border border-warning' : 'bg-light'}`}>
                                        <strong>{feature.feature}:</strong> {feature.value}
                                        <div className="d-flex justify-content-between align-items-center mt-1">
                                          <Badge bg={feature.frequency_rank === 1 ? 'warning' : 'info'} size="sm">
                                            Rank #{feature.frequency_rank}
                                          </Badge>
                                          <small className="text-muted">{feature.percentage}%</small>
                                        </div>
                                      </div>
                                    </Col>
                                  ))}
                                </Row>
                              </Card.Body>
                            </Card>
                          </>
                        )}
                      </>
                    ) : (
                      <Alert variant="warning">
                        No sufficient data for {explainabilityData.class_distribution[classKey].label.toLowerCase()} analysis.
                      </Alert>
                    )}
                  </div>
                ))}
              </Card.Body>
            </Card>
          )}
        </Card.Body>
      </Card>
    );
  };

  return (
    <Container fluid className="upload-container">
      <button
        className="theme-toggle"
        onClick={() => setDarkMode((dm) => !dm)}
        title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      >
        {darkMode ? <FiMoon /> : <FiSun />}
      </button>
      <Row className="justify-content-center">
        <Col lg={10}>
          <div className="text-center mb-4">
            <h1 className="display-4 text-primary">ML Prediction System</h1>
            <p className="lead text-muted">
              Upload your CSV or Excel files to get predictions using our pre-trained models
            </p>
          </div>

          {error && (
            <Alert variant="danger" dismissible onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {/* File Upload Section */}
          {!fileData && (
            <Card>
              <Card.Body>
                <div
                  {...getRootProps()}
                  className={`upload-area ${isDragActive ? 'dragover' : ''}`}
                >
                  <input {...getInputProps()} />
                  <FiUpload size={48} className="text-primary mb-3" />
                  <h4>Drop your file here, or click to select</h4>
                  <p className="text-muted" style={{ color: 'var(--bs.body-color)' }}>
                    Supports CSV and Excel files (max 100,000 rows) - Files are automatically converted to CSV for faster processing
                  </p>
                  <p className="text-muted small" style={{ color: 'var(--bs.body-color)' }}>
                    File must contain a "projectid" column
                  </p>
                </div>
              </Card.Body>
            </Card>
          )}

          {/* Loading Spinner */}
          {loading && (
            <div className="loading-spinner">
              <div className="text-center">
                <Spinner animation="border" role="status" className="mb-3">
                  <span className="visually-hidden">Loading...</span>
                </Spinner>
                <p>Processing your file...</p>
              </div>
            </div>
          )}

          {/* File Preview Section */}
          {fileData && !loading && showPreview && (
            <>
              <Card className="mb-4">
                <Card.Header>
                  <h5 className="mb-0">File Information</h5>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col md={6}>
                      <p><strong>Filename:</strong> {fileData.originalName || fileData.filename}</p>
                    </Col>
                    <Col md={6}>
                      <Button
                        variant="primary"
                        onClick={generatePredictions}
                        disabled={loading || !fileData}
                        className="me-2"
                      >
                        <FiDatabase className="me-2" />
                        Generate Predictions
                      </Button>
                      <Button variant="outline-secondary" onClick={cleanup}>
                        Upload New File
                      </Button>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>
              
              {/* Project Filters */}
              <ProjectFilters
                projects={availableProjects}
                selectedProject={selectedProject}
                onSelect={handleProjectFilter}
                onReset={handleFilterReset}
                disabled={loading}
                className="mb-4"
              />
              
              {/* Data Statistics */}
              <DataStats
                data={currentData}
                title={selectedProject ? `Project ${selectedProject} Statistics` : "Full Dataset Statistics"}
                className="mb-4"
              />
              
              {renderPreviewControls()}
              {/* Preview Table */}
              {showPreview && (
                (!previewRows || previewRows.length === 0) ? (
                  <Alert variant="warning">No preview data available for this file.</Alert>
                ) : (
                  renderPreviewTable(previewRows, `Data Preview ${selectedProject ? `(Project ${selectedProject})` : '(All Projects)'}`)
                )
              )}
            </>
          )}

          {/* Prediction Results Section */}
          {predictions && !loading && (
            <>
              <Card className="mb-4">
                <Card.Header>
                  <h5 className="mb-0">Prediction Results Summary</h5>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col md={8}>
                      <p><strong>Rows Processed:</strong> {predictions.stats?.rows_processed ?? '-'}</p>
                      <p><strong>Prediction Distribution:</strong> 
                        {predictions.stats?.prediction_distribution ? 
                          (predictions.stats.prediction_distribution['1'] > 0 ? 
                            `${predictions.stats.prediction_distribution['1']} potential customers, ` : '') +
                          (predictions.stats.prediction_distribution['0'] > 0 ? 
                            `${predictions.stats.prediction_distribution['0']} not potential customers` : '')
                        : '-'}
                      </p>
                    </Col>
                    <Col md={4} className="d-flex align-items-center justify-content-end">
                      <Button variant="outline-secondary" onClick={cleanup}>
                        Upload New File
                      </Button>
                    </Col>
                  </Row>
                </Card.Body>
              </Card>

              {/* Prediction Statistics */}
              {renderPredictionStats()}

              {/* Project Filters for Prediction Results */}
              <ProjectFilters
                projects={availableProjects}
                selectedProject={selectedProject}
                onSelect={handleProjectFilter}
                onReset={handleFilterReset}
                disabled={loading}
                className="mb-4"
              />
              
              {/* Prediction Data Statistics */}
              <DataStats
                data={currentData}
                title={selectedProject ? `Project ${selectedProject} Prediction Statistics` : "Full Prediction Statistics"}
                className="mb-4"
              />
              
              {renderPreviewControls()}
              
              {/* Final Prediction Preview */}
              <h4 className="mt-3 mb-3" style={{ color: '#0d6efd' }}>Final Prediction Preview: Prediction Results</h4>
              {renderPreviewTable(previewRows, `Prediction Results ${selectedProject ? `(Project ${selectedProject})` : '(All Projects)'}`)}

              {/* EXPLAINABILITY ANALYSIS SECTION - ADD THIS HERE */}
              {renderExplainabilitySection()}

              {/* Export Results */}
              <Card>
                <Card.Header>
                  <h5 className="mb-0">
                    <FiDownload className="me-2" />
                    Export Results
                  </h5>
                </Card.Header>
                <Card.Body>
                  <div className="d-flex flex-wrap gap-2">
                    <Button
                      variant="outline-primary"
                      onClick={() => exportResults('csv')}
                      disabled={exportLoading || loading}
                      className="btn-export d-flex align-items-center"
                    >
                      <FiDatabase className="me-2" />
                      {exportLoading && exportFormat === 'CSV' ? (
                        <>
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            role="status"
                            aria-hidden="true"
                            className="me-2"
                          />
                          Exporting...
                        </>
                      ) : (
                        'Export as CSV'
                      )}
                    </Button>
                    <Button
                      variant="outline-success"
                      onClick={() => exportResults('xlsx')}
                      disabled={exportLoading || loading}
                      className="btn-export d-flex align-items-center"
                    >
                      <FiGrid className="me-2" />
                      {exportLoading && exportFormat === 'XLSX' ? (
                        <>
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            role="status"
                            aria-hidden="true"
                            className="me-2"
                          />
                          Exporting...
                        </>
                      ) : (
                        'Export as Excel'
                      )}
                    </Button>
                    <Button
                      variant="outline-info"
                      onClick={() => exportResults('json')}
                      disabled={exportLoading || loading}
                      className="btn-export d-flex align-items-center"
                    >
                      <FiCode className="me-2" />
                      {exportLoading && exportFormat === 'JSON' ? (
                        <>
                          <Spinner
                            as="span"
                            animation="border"
                            size="sm"
                            role="status"
                            aria-hidden="true"
                            className="me-2"
                          />
                          Exporting...
                        </>
                      ) : (
                        'Export as JSON'
                      )}
                    </Button>
                  </div>
                  {exportLoading && (
                    <div className="mt-3">
                      <small className="text-muted">
                        <FiClock className="me-1" />
                        Export in progress... Please wait while we prepare your {exportFormat} file.
                      </small>
                    </div>
                  )}
                </Card.Body>
              </Card>

              {/* Upload New File button at the bottom */}
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem', marginBottom: '1rem' }}>
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={cleanup}
                  style={{ minWidth: 220, fontWeight: 600 }}
                >
                  Upload New File
                </Button>
              </div>
            </>
          )}

          {/* File Retention and Storage Info */}
          {fileInfo && (
            <>
              {renderFileRetentionInfo()}
              {renderStorageStats()}
            </>
          )}
        </Col>
      </Row>
      
      {/* Export Progress Modal */}
      {renderExportModal()}
    </Container>
  );
}

export default App;