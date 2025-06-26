import React, { useMemo } from 'react';
import { Card, Row, Col, Badge } from 'react-bootstrap';
import { FiDatabase, FiColumns, FiInfo } from 'react-icons/fi';

const DataStats = ({ data, title = "Data Statistics", className = '' }) => {
  const stats = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        totalRows: 0,
        totalColumns: 0,
        numericColumns: 0,
        categoricalColumns: 0,
        missingValues: 0,
        uniqueValues: {},
        columnTypes: {}
      };
    }

    const columns = Object.keys(data[0]);
    const totalRows = data.length;
    const totalColumns = columns.length;
    
    let numericColumns = 0;
    let categoricalColumns = 0;
    let missingValues = 0;
    const uniqueValues = {};
    const columnTypes = {};

    columns.forEach(col => {
      const values = data.map(row => row[col]).filter(val => val !== null && val !== undefined && val !== '');
      const uniqueCount = new Set(values).size;
      uniqueValues[col] = uniqueCount;
      
      // Count missing values
      const missingCount = data.length - values.length;
      missingValues += missingCount;

      // Determine column type
      const sampleValues = values.slice(0, 10);
      const isNumeric = sampleValues.every(val => {
        if (typeof val === 'number') return true;
        if (typeof val === 'string') {
          const num = parseFloat(val);
          return !isNaN(num) && val.trim() !== '';
        }
        return false;
      });

      if (isNumeric) {
        numericColumns++;
        columnTypes[col] = 'numeric';
      } else {
        categoricalColumns++;
        columnTypes[col] = 'categorical';
      }
    });

    return {
      totalRows,
      totalColumns,
      numericColumns,
      categoricalColumns,
      missingValues,
      uniqueValues,
      columnTypes
    };
  }, [data]);

  if (!data || data.length === 0) {
    return (
      <Card className={`data-stats ${className}`}>
        <Card.Header>
          <h6 className="mb-0">
            <FiInfo className="me-2" />
            {title}
          </h6>
        </Card.Header>
        <Card.Body>
          <p className="text-muted mb-0">No data available</p>
        </Card.Body>
      </Card>
    );
  }

  return (
    <Card className={`data-stats ${className}`}>
      <Card.Header>
        <h6 className="mb-0">
          <FiDatabase className="me-2" />
          {title}
        </h6>
      </Card.Header>
      <Card.Body>
        <Row>
          <Col md={3} sm={6} className="mb-3">
            <div className="text-center">
              <div className="stats-number text-primary">{stats.totalRows.toLocaleString()}</div>
              <div className="text-muted">Total Rows</div>
            </div>
          </Col>
          <Col md={3} sm={6} className="mb-3">
            <div className="text-center">
              <div className="stats-number text-info">{stats.totalColumns}</div>
              <div className="text-muted">Total Columns</div>
            </div>
          </Col>
          <Col md={3} sm={6} className="mb-3">
            <div className="text-center">
              <div className="stats-number text-success">{stats.numericColumns}</div>
              <div className="text-muted">Numeric Columns</div>
            </div>
          </Col>
          <Col md={3} sm={6} className="mb-3">
            <div className="text-center">
              <div className="stats-number text-warning">{stats.categoricalColumns}</div>
              <div className="text-muted">Categorical Columns</div>
            </div>
          </Col>
        </Row>

        {stats.missingValues > 0 && (
          <Row className="mt-3">
            <Col>
              <div className="alert alert-warning py-2 mb-0">
                <FiInfo className="me-2" />
                <strong>{stats.missingValues.toLocaleString()}</strong> missing values detected
              </div>
            </Col>
          </Row>
        )}

        <Row className="mt-3">
          <Col>
            <h6 className="mb-2">
              <FiColumns className="me-2" />
              Column Overview
            </h6>
            <div className="column-stats">
              {Object.keys(stats.uniqueValues).slice(0, 5).map(col => (
                <Badge 
                  key={col}
                  bg={stats.columnTypes[col] === 'numeric' ? 'success' : 'info'}
                  className="me-2 mb-1"
                >
                  {col}: {stats.uniqueValues[col]} unique
                </Badge>
              ))}
              {Object.keys(stats.uniqueValues).length > 5 && (
                <Badge bg="secondary" className="me-2 mb-1">
                  +{Object.keys(stats.uniqueValues).length - 5} more
                </Badge>
              )}
            </div>
          </Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

export default DataStats; 