import React, { useState, useEffect } from 'react';
import { Card, Button, Spinner, Alert, Row, Col, Table, Container, Badge, Form, Nav } from 'react-bootstrap';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { FiInfo, FiBarChart2, FiPieChart, FiDownload, FiArrowLeft, FiRefreshCw } from 'react-icons/fi';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

// API URL from environment or default to localhost
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const InterpretMLAnalysis = ({ fileInfo, onBack, selectedProject, onSelectProject, availableProjects }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [currentProject, setCurrentProject] = useState(selectedProject);
  const [chartType, setChartType] = useState('bar'); // 'bar' or 'pie'
  const [analysisMethod, setAnalysisMethod] = useState('shap'); // 'shap' or 'interpret'

  // Color palettes for charts
  const barColors = [
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 99, 132, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(199, 199, 199, 0.8)',
    'rgba(83, 102, 255, 0.8)',
    'rgba(40, 159, 64, 0.8)',
    'rgba(210, 199, 199, 0.8)',
  ];

  const pieColors = [
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 99, 132, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(199, 199, 199, 0.8)',
    'rgba(83, 102, 255, 0.8)',
    'rgba(40, 159, 64, 0.8)',
    'rgba(210, 199, 199, 0.8)',
  ];

  useEffect(() => {
    if (selectedProject) {
      setCurrentProject(selectedProject);
    }
  }, [selectedProject]);

  useEffect(() => {
    if (currentProject && fileInfo) {
      fetchAnalysis();
    }
  }, [currentProject, fileInfo, analysisMethod]);

  const fetchAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      let url;
      if (analysisMethod === 'interpret') {
        url = `${API_URL}/api/interpret_ml/${fileInfo.filename}?project_id=${currentProject}`;
      } else {
        url = `${API_URL}/api/feature_importance/${fileInfo.filename}?project_id=${currentProject}`;
      }
      
      const response = await axios.get(url);
      console.log(`Fetched ${analysisMethod} data:`, response.data);
      
      // Check data structure and provide debugging info
      if (response.data && response.data.analysis) {
        if (analysisMethod === 'interpret' && !response.data.analysis.interpret_data) {
          console.warn('InterpretML data missing interpret_data field:', response.data.analysis);
        }
        if (!response.data.analysis.shap_data) {
          console.warn('Analysis data missing shap_data field:', response.data.analysis);
        }
      }
      
      setAnalysisData(response.data);
    } catch (err) {
      console.error('Error fetching analysis:', err);
      setError(`Failed to load analysis: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleProjectChange = (e) => {
    const newProjectId = parseInt(e.target.value);
    setCurrentProject(newProjectId);
    if (onSelectProject) {
      onSelectProject(newProjectId);
    }
  };

  const toggleChartType = () => {
    setChartType(chartType === 'bar' ? 'pie' : 'bar');
  };

  const toggleAnalysisMethod = (method) => {
    setAnalysisMethod(method);
  };

  // Helper function to get the appropriate data field based on analysis method
  const getDataField = () => {
    if (!analysisData || !analysisData.analysis) return null;
    
    if (analysisMethod === 'interpret') {
      // For InterpretML, try interpret_data first, then fall back to shap_data
      return analysisData.analysis.interpret_data || analysisData.analysis.shap_data;
    } else {
      // For SHAP, use shap_data
      return analysisData.analysis.shap_data;
    }
  };

  const renderChart = () => {
    const dataField = getDataField();
    if (!dataField) return null;

    const { feature_names, importance_pct } = dataField;
    
    // Limit to top 10 features
    const topFeatures = feature_names.slice(0, 10);
    const topImportance = importance_pct.slice(0, 10);

    if (chartType === 'bar') {
      const barData = {
        labels: topFeatures,
        datasets: [
          {
            label: 'Feature Importance (%)',
            data: topImportance,
            backgroundColor: barColors,
            borderColor: barColors.map(c => c.replace('0.8', '1')),
            borderWidth: 1,
          },
        ],
      };

      return (
        <Bar 
          data={barData} 
          options={{
            indexAxis: 'y',
            responsive: true,
            plugins: {
              legend: { position: 'top' },
              title: { 
                display: true, 
                text: `Feature Importance Analysis ${analysisMethod === 'interpret' ? '(InterpretML)' : '(SHAP)'}` 
              },
              tooltip: {
                callbacks: {
                  label: function(context) {
                    return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
                  }
                }
              }
            },
            scales: {
              x: {
                beginAtZero: true,
                title: {
                  display: true,
                  text: 'Importance (%)'
                }
              }
            }
          }} 
        />
      );
    } else {
      const pieData = {
        labels: topFeatures,
        datasets: [
          {
            label: 'Feature Importance (%)',
            data: topImportance,
            backgroundColor: pieColors,
            borderColor: pieColors.map(c => c.replace('0.8', '1')),
            borderWidth: 1,
          },
        ],
      };

      return (
        <Pie 
          data={pieData} 
          options={{
            responsive: true,
            plugins: {
              legend: { position: 'right' },
              title: { 
                display: true, 
                text: `Feature Importance Analysis ${analysisMethod === 'interpret' ? '(InterpretML)' : '(SHAP)'}` 
              },
              tooltip: {
                callbacks: {
                  label: function(context) {
                    return `${context.label}: ${context.raw.toFixed(2)}%`;
                  }
                }
              }
            }
          }} 
        />
      );
    }
  };

  const renderFeatureTable = () => {
    const dataField = getDataField();
    if (!dataField) return null;

    const { feature_names, importance_pct } = dataField;
    
    return (
      <Table striped bordered hover responsive className="mt-3">
        <thead>
          <tr>
            <th>#</th>
            <th>Feature</th>
            <th>Importance (%)</th>
          </tr>
        </thead>
        <tbody>
          {feature_names.map((feature, idx) => (
            <tr key={idx}>
              <td>{idx + 1}</td>
              <td>{feature}</td>
              <td>{importance_pct[idx].toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </Table>
    );
  };

  const renderModelInfo = () => {
    const dataField = getDataField();
    if (!dataField) return null;

    const { model_type, note } = dataField;
    const { status, method } = analysisData.analysis;

    return (
      <Card className="mt-3">
        <Card.Header>
          <h5>Model Information</h5>
        </Card.Header>
        <Card.Body>
          <Row>
            <Col md={6}>
              <p><strong>Project ID:</strong> {currentProject}</p>
              <p><strong>Model Type:</strong> {model_type || 'Unknown'}</p>
              <p><strong>Analysis Status:</strong> <Badge bg={status === 'success' ? 'success' : status === 'warning' ? 'warning' : 'danger'}>{status}</Badge></p>
            </Col>
            <Col md={6}>
              <p><strong>Analysis Method:</strong> {analysisMethod === 'interpret' ? 'InterpretML' : 'SHAP'}</p>
              <p><strong>Implementation:</strong> {method || analysisMethod}</p>
              {note && <p><strong>Note:</strong> {note}</p>}
            </Col>
          </Row>
        </Card.Body>
      </Card>
    );
  };

  return (
    <Container fluid>
      <Card>
        <Card.Header>
          <div className="d-flex justify-content-between align-items-center">
            <h4><FiInfo /> Advanced Feature Importance Analysis</h4>
            <Button variant="outline-secondary" onClick={onBack}>
              <FiArrowLeft /> Back to Predictions
            </Button>
          </div>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={4}>
              <Form.Group>
                <Form.Label>Select Project:</Form.Label>
                <Form.Control as="select" value={currentProject || ''} onChange={handleProjectChange}>
                  <option value="">Select a project...</option>
                  {availableProjects.map(project => (
                    <option key={project} value={project}>Project {project}</option>
                  ))}
                </Form.Control>
              </Form.Group>
            </Col>
            <Col md={4}>
              <Form.Label>Analysis Method:</Form.Label>
              <Nav variant="pills">
                <Nav.Item>
                  <Nav.Link 
                    active={analysisMethod === 'shap'}
                    onClick={() => toggleAnalysisMethod('shap')}
                  >
                    SHAP
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link 
                    active={analysisMethod === 'interpret'}
                    onClick={() => toggleAnalysisMethod('interpret')}
                  >
                    InterpretML
                  </Nav.Link>
                </Nav.Item>
              </Nav>
            </Col>
            <Col md={4} className="d-flex align-items-end">
              <Button 
                variant="outline-primary" 
                onClick={toggleChartType} 
                className="me-2"
              >
                {chartType === 'bar' ? <><FiPieChart /> Switch to Pie Chart</> : <><FiBarChart2 /> Switch to Bar Chart</>}
              </Button>
              <Button 
                variant="outline-success" 
                onClick={fetchAnalysis}
              >
                <FiRefreshCw /> Refresh
              </Button>
            </Col>
          </Row>
          
          {error && (
            <Alert variant="danger">{error}</Alert>
          )}
          
          {loading ? (
            <div className="text-center my-5">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
              <p className="mt-2">Loading {analysisMethod === 'interpret' ? 'InterpretML' : 'SHAP'} analysis...</p>
            </div>
          ) : analysisData ? (
            <>
              <Row>
                <Col lg={6}>
                  <Card>
                    <Card.Header>
                      <h5>Feature Importance Visualization</h5>
                    </Card.Header>
                    <Card.Body>
                      {renderChart()}
                    </Card.Body>
                  </Card>
                </Col>
                <Col lg={6}>
                  {renderModelInfo()}
                  <Card className="mt-3">
                    <Card.Header>
                      <h5>Feature Importance Ranking</h5>
                    </Card.Header>
                    <Card.Body style={{maxHeight: '400px', overflowY: 'auto'}}>
                      {renderFeatureTable()}
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
            </>
          ) : !currentProject ? (
            <Alert variant="info">Please select a project to view feature importance analysis.</Alert>
          ) : (
            <Alert variant="info">No analysis data available. Make sure you've processed predictions for this project.</Alert>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
};

export default InterpretMLAnalysis;
