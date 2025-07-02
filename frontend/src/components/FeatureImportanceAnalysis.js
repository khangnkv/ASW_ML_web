import React, { useState, useEffect } from 'react';
import { Card, Button, Spinner, Alert, Row, Col, Table, Container, Badge, Form } from 'react-bootstrap';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { FiInfo, FiBarChart2, FiPieChart, FiDownload, FiArrowLeft } from 'react-icons/fi';

// Register ChartJS components
ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend);

const FeatureImportanceAnalysis = ({ fileInfo, onBack, selectedProject, onSelectProject, availableProjects }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [currentProject, setCurrentProject] = useState(selectedProject);
  const [chartType, setChartType] = useState('bar'); // 'bar' or 'pie'

  // Color palettes for charts
  const barColors = [
    'rgba(54, 162, 235, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    'rgba(255, 99, 132, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(199, 199, 199, 0.8)',
    'rgba(83, 102, 255, 0.8)',
    'rgba(40, 159, 163, 0.8)',
    'rgba(210, 105, 30, 0.8)'
  ];

  const pieColors = [
    'rgba(54, 162, 235, 0.7)',
    'rgba(75, 192, 192, 0.7)',
    'rgba(255, 206, 86, 0.7)',
    'rgba(255, 99, 132, 0.7)',
    'rgba(153, 102, 255, 0.7)',
    'rgba(255, 159, 64, 0.7)',
    'rgba(199, 199, 199, 0.7)',
    'rgba(83, 102, 255, 0.7)',
    'rgba(40, 159, 163, 0.7)',
    'rgba(210, 105, 30, 0.7)'
  ];

  useEffect(() => {
    if (fileInfo && currentProject) {
      loadAnalysisData();
    }
  }, [fileInfo, currentProject]);

  const loadAnalysisData = async () => {
    if (!fileInfo?.filename) {
      setError('No file selected');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const url = `${API_URL}/api/shap_visualization/${fileInfo.filename}?project_id=${currentProject}`;

      const response = await axios.get(url);
      setAnalysisData(response.data);
    } catch (error) {
      console.error('Error loading feature importance data:', error);
      setError(error.response?.data?.error || 'Error loading analysis data');
    } finally {
      setLoading(false);
    }
  };

  const handleProjectChange = (e) => {
    const projectId = e.target.value;
    setCurrentProject(projectId);
    if (onSelectProject) {
      onSelectProject(projectId);
    }
  };

  // Format chart data for bar chart
  const getBarChartData = () => {
    if (!analysisData || !analysisData.feature_names) return null;

    return {
      labels: analysisData.feature_names,
      datasets: [
        {
          label: 'Feature Importance (%)',
          data: analysisData.importance_pct,
          backgroundColor: barColors,
          borderColor: barColors.map(color => color.replace('0.8', '1')),
          borderWidth: 1
        }
      ]
    };
  };

  // Format chart data for pie chart
  const getPieChartData = () => {
    if (!analysisData || !analysisData.feature_names) return null;

    return {
      labels: analysisData.feature_names,
      datasets: [
        {
          data: analysisData.importance_pct,
          backgroundColor: pieColors,
          borderColor: pieColors.map(color => color.replace('0.7', '1')),
          borderWidth: 1
        }
      ]
    };
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Importance (%)'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Features'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `Feature Importance for Project ${currentProject}`
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const label = context.label || '';
            const value = context.raw || 0;
            return `${label}: ${value.toFixed(2)}%`;
          }
        }
      }
    }
  };

  // Pie chart options
  const pieOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
      },
      title: {
        display: true,
        text: `Feature Importance for Project ${currentProject}`
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const label = context.label || '';
            const value = context.raw || 0;
            return `${label}: ${value.toFixed(2)}%`;
          }
        }
      }
    }
  };

  // Download chart as image
  const downloadChart = () => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = `feature_importance_project_${currentProject}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  return (
    <Container fluid>
      <Card className="mb-4">
        <Card.Header className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">
            <FiInfo className="me-2" />
            Feature Importance Analysis
          </h5>
          <Button variant="outline-secondary" size="sm" onClick={onBack}>
            <FiArrowLeft /> Back to Predictions
          </Button>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={6}>
              <Form.Group>
                <Form.Label><strong>Select Project:</strong></Form.Label>
                <Form.Select 
                  value={currentProject || ''} 
                  onChange={handleProjectChange}
                  className="mb-3"
                >
                  <option value="">-- Select Project --</option>
                  {availableProjects.map(project => (
                    <option key={project} value={project}>Project {project}</option>
                  ))}
                </Form.Select>
              </Form.Group>
            </Col>
            <Col md={6}>
              <Form.Group>
                <Form.Label><strong>Chart Type:</strong></Form.Label>
                <div>
                  <Button 
                    variant={chartType === 'bar' ? 'primary' : 'outline-primary'}
                    className="me-2"
                    onClick={() => setChartType('bar')}
                  >
                    <FiBarChart2 className="me-1" /> Bar Chart
                  </Button>
                  <Button 
                    variant={chartType === 'pie' ? 'primary' : 'outline-primary'}
                    onClick={() => setChartType('pie')}
                  >
                    <FiPieChart className="me-1" /> Pie Chart
                  </Button>
                </div>
              </Form.Group>
            </Col>
          </Row>

          {loading && (
            <div className="text-center my-4">
              <Spinner animation="border" role="status">
                <span className="visually-hidden">Loading...</span>
              </Spinner>
              <p className="mt-2">Loading feature importance analysis...</p>
            </div>
          )}

          {error && (
            <Alert variant="danger" className="mb-4">
              {error}
            </Alert>
          )}

          {analysisData && !loading && (
            <>
              <Card className="mb-4">
                <Card.Header>
                  <h6 className="mb-0">
                    <Badge bg="info" className="me-2">Project {currentProject}</Badge>
                    {analysisData.model_type && (
                      <Badge bg="secondary" className="me-2">Model: {analysisData.model_type}</Badge>
                    )}
                  </h6>
                </Card.Header>
                <Card.Body>
                  <div className="d-flex justify-content-end mb-3">
                    <Button variant="outline-secondary" size="sm" onClick={downloadChart}>
                      <FiDownload className="me-1" /> Download Chart
                    </Button>
                  </div>
                  <div style={{ height: '400px' }}>
                    {chartType === 'bar' ? (
                      <Bar data={getBarChartData()} options={chartOptions} />
                    ) : (
                      <Pie data={getPieChartData()} options={pieOptions} />
                    )}
                  </div>
                </Card.Body>
              </Card>

              <Card className="mb-4">
                <Card.Header>
                  <h6 className="mb-0">Feature Importance Details</h6>
                </Card.Header>
                <Card.Body className="p-0">
                  <Table responsive striped hover className="mb-0">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Feature Name</th>
                        <th>Importance (%)</th>
                        <th>Normalized Impact</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisData.feature_names.map((feature, index) => (
                        <tr key={index}>
                          <td>{index + 1}</td>
                          <td><strong>{feature}</strong></td>
                          <td>{analysisData.importance_pct[index].toFixed(2)}%</td>
                          <td>
                            <div className="progress" style={{ height: '25px' }}>
                              <div 
                                className="progress-bar" 
                                role="progressbar" 
                                style={{ 
                                  width: `${analysisData.importance_pct[index]}%`,
                                  backgroundColor: barColors[index % barColors.length]
                                }}
                                aria-valuenow={analysisData.importance_pct[index]}
                                aria-valuemin="0" 
                                aria-valuemax="100"
                              >
                                {analysisData.importance_pct[index].toFixed(1)}%
                              </div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                </Card.Body>
              </Card>
            </>
          )}

          {!analysisData && !loading && !error && currentProject && (
            <Alert variant="info">
              Select a project to view feature importance analysis
            </Alert>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
};

export default FeatureImportanceAnalysis;
