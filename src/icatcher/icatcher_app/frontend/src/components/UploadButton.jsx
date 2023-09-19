import React, { useRef } from 'react';

function UploadButton (props) {
 
  const { handleInput } = props;
  const hiddenInputElement = useRef(null);
  
  const handleClick = (event) => {
    hiddenInputElement.current.click();
  };

  const handleChange = (event) => {
    handleInput(event);
    console.log("EEE", hiddenInputElement.current)
    console.log("SSS", event)

  };
return (
    <React.Fragment>
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
        <div />
    </React.Fragment>
  );
}

export default UploadButton;