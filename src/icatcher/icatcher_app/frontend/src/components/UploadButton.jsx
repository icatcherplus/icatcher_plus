import React, { useRef, useState } from 'react';

function UploadButton (props) {
 
  const { handleInput } = props;
  const hiddenInputElement = useRef(null);
  const [ inputSize, setInputSize ] = useState(null);
  
  const handleClick = (event) => {
    hiddenInputElement.current.click();
  };

  const handleChange = (event) => {
    event.target.files ? setInputSize(event.target.files.length) : setInputSize(null)
    handleInput(event);
  };
return (
    <div style={{display:'flex', gap: '10px'}}>
      <button className="button-upload" onClick={handleClick}>
        Upload output directory
      </button>
      <input 
        type="file" 
        id="fileInput" 
        onChange={handleChange}
        ref={hiddenInputElement}
        webkitdirectory=""
        style={{display: 'none'}}
      />
      <div>
        { inputSize == null 
          ? ''
          : `${inputSize} files`
        }
      </div>
    </div>
  );
}

export default UploadButton;