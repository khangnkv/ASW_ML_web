import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Alert, Spinner } from 'react-bootstrap';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { saveAs } from 'file-saver';
import { FiUpload, FiDownload, FiDatabase, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [fileData, setFileData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch available models on component mount
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get('/api/models');
      // We'll use this data later if needed
      console.log('Available models:', response.data.available_models);
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setFileData(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Error uploading file');
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

  const generatePredictions = async () => {
    if (!fileData) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/predict', {
        filename: fileData.filename,
      });

      setPredictions(response.data);
    } catch (error) {
      setError(error.response?.data?.error || 'Error generating predictions');
    } finally {
      setLoading(false);
    }
  };

  const exportResults = async (format) => {
    if (!predictions) return;

    try {
      if (format === 'json') {
        const response = await axios.get(`/api/export/${format}/${predictions.results_filename}`);
        const blob = new Blob([JSON.stringify(response.data, null, 2)], {
          type: 'application/json',
        });
        saveAs(blob, `predictions_${fileData.filename.split('.')[0]}.json`);
      } else {
        const response = await axios.get(`/api/export/${format}/${predictions.results_filename}`, {
          responseType: 'blob',
        });
        const extension = format === 'csv' ? 'csv' : 'xlsx';
        saveAs(response.data, `predictions_${fileData.filename.split('.')[0]}.${extension}`);
      }
    } catch (error) {
      setError('Error exporting results');
    }
  };

  const cleanup = async () => {
    if (!fileData) return;

    try {
      await axios.post('/api/cleanup', {
        filename: fileData.filename,
      });
      setFileData(null);
      setPredictions(null);
      setError(null);
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  };

  const renderPreviewTable = (data, title) => {
    if (!data || data.length === 0) return null;

    let columns = Object.keys(data[0]);
    // Ensure has_booked_prediction is last
    if (columns.includes('has_booked_prediction')) {
      columns = columns.filter(c => c !== 'has_booked_prediction').concat(['has_booked_prediction']);
    }

    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">{title}</h5>
        </Card.Header>
        <Card.Body className="p-0">
          <div className="table-responsive">
            <table className="table table-striped table-hover mb-0">
              <thead className="table-primary">
                <tr>
                  {columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.map((row, index) => (
                  <tr key={index}>
                    {columns.map((column) => (
                      <td key={column}>
                        {typeof row[column] === 'boolean' 
                          ? (row[column] ? 'Yes' : 'No')
                          : row[column]?.toString() || ''}
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

  const renderModelStatus = () => {
    if (!fileData) return null;

    const { available_models, missing_models, unique_projects } = fileData;
    const totalProjects = unique_projects.length;
    const availableCount = available_models.length;
    const missingCount = missing_models.length;

    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">
            <FiDatabase className="me-2" />
            Model Availability
          </h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={4}>
              <div className="text-center">
                <div className="stats-number text-success">{availableCount}</div>
                <div className="text-muted">Available Models</div>
              </div>
            </Col>
            <Col md={4}>
              <div className="text-center">
                <div className="stats-number text-danger">{missingCount}</div>
                <div className="text-muted">Missing Models</div>
              </div>
            </Col>
            <Col md={4}>
              <div className="text-center">
                <div className="stats-number text-primary">{totalProjects}</div>
                <div className="text-muted">Total Projects</div>
              </div>
            </Col>
          </Row>
          
          <div className="mt-3">
            <h6>Project Details:</h6>
            <div className="d-flex flex-wrap gap-2">
              {unique_projects.map((projectId) => {
                const isAvailable = available_models.includes(projectId);
                return (
                  <span
                    key={projectId}
                    className={`model-status ${isAvailable ? 'model-available' : 'model-missing'}`}
                  >
                    {isAvailable ? <FiCheckCircle className="me-1" /> : <FiXCircle className="me-1" />}
                    Project {projectId}
                  </span>
                );
              })}
            </div>
          </div>
        </Card.Body>
      </Card>
    );
  };

  const renderPredictionStats = () => {
    if (!predictions) return null;

    const { prediction_stats } = predictions;

    return (
      <Card className="mb-4">
        <Card.Header>
          <h5 className="mb-0">Prediction Results</h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-primary">{prediction_stats.total_predictions}</div>
                <div className="text-muted">Total Rows</div>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-success">{prediction_stats.successful_predictions}</div>
                <div className="text-muted">Successful</div>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-warning">{prediction_stats.failed_predictions}</div>
                <div className="text-muted">Failed</div>
              </div>
            </Col>
            <Col md={3}>
              <div className="text-center">
                <div className="stats-number text-info">{prediction_stats.prediction_rate}%</div>
                <div className="text-muted">Success Rate</div>
              </div>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    );
  };

  return (
    <Container fluid className="upload-container">
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
                  <p className="text-muted">
                    Supports CSV and Excel files (max 100,000 rows)
                  </p>
                  <p className="text-muted small">
                    File must contain a "projectid" column
                  </p>
                </div>
              </Card.Body>
            </Card>
          )}

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

          {fileData && !loading && (
            <>
              <Card className="mb-4">
                <Card.Header>
                  <h5 className="mb-0">File Information</h5>
                </Card.Header>
                <Card.Body>
                  <Row>
                    <Col md={6}>
                      <p><strong>Filename:</strong> {fileData.filename}</p>
                      <p><strong>Total Rows:</strong> {fileData.total_rows.toLocaleString()}</p>
                      <p><strong>Columns:</strong> {fileData.columns.length}</p>
                    </Col>
                    <Col md={6}>
                      <Button
                        variant="primary"
                        onClick={generatePredictions}
                        disabled={loading}
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

              {renderModelStatus()}

              {renderPreviewTable(fileData.preview_data, 'File Preview (First & Last 5 Rows)')}
            </>
          )}

          {predictions && !loading && (
            <>
              {renderPredictionStats()}

              {renderPreviewTable(predictions.preview_data, 'Prediction Results Preview')}

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
                      className="btn-export"
                    >
                      Export as CSV
                    </Button>
                    <Button
                      variant="outline-success"
                      onClick={() => exportResults('xlsx')}
                      className="btn-export"
                    >
                      Export as Excel
                    </Button>
                    <Button
                      variant="outline-info"
                      onClick={() => exportResults('json')}
                      className="btn-export"
                    >
                      Export as JSON
                    </Button>
                  </div>
                </Card.Body>
              </Card>
            </>
          )}
        </Col>
      </Row>
    </Container>
  );
}

export default App; 