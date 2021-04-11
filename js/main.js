/**
 * Generate a scene object with a background color
 **/
function getScene() {
  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0xFFFFFF);
  return scene;
}

/**
* Generate the camera to be used in the scene. Camera args:
*   [0] field of view: identifies the portion of the scene
*     visible at any time (in degrees)
*   [1] aspect ratio: identifies the aspect ratio of the
*     scene in width/height
*   [2] near clipping plane: objects closer than the near
*     clipping plane are culled from the scene
*   [3] far clipping plane: objects farther than the far
*     clipping plane are culled from the scene
**/

function getCamera() {
  var aspectRatio = window.innerWidth / window.innerHeight;
  var camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
  camera.position.set(x= 0.0938121251078376, y= -2.394539599542562, z= 0.3411419154271028);
  return camera;
}

/**
* Generate the light to be used in the scene. Light args:
*   [0]: Hexadecimal color of the light
*   [1]: Numeric value of the light's strength/intensity
*   [2]: The distance from the light where the intensity is 0
* @param {obj} scene: the current scene object
**/

function getLight(scene) {
  var light = new THREE.PointLight(0xffffff, 1, 0);
  light.position.set(1, 1, 1);
  scene.add(light);

  var ambientLight = new THREE.AmbientLight(0x111111);
  scene.add(ambientLight);
  return light;
}

/**
  * Generate the renderer to be used in the scene
  **/

function getRenderer() {
  // Create the canvas with a renderer
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  // Add support for retina displays
  renderer.setPixelRatio(window.devicePixelRatio);
  // Specify the size of the canvas
  renderer.setSize(window.innerWidth, window.innerHeight);
  // Add the canvas to the DOM
  document.body.appendChild(renderer.domElement);
  return renderer;
}

/**
  * Generate the controls to be used in the scene
  * @param {obj} camera: the three.js camera for the scene
  * @param {obj} renderer: the three.js renderer for the scene
  **/

function getControls(camera, renderer) {
  var controls = new THREE.TrackballControls(camera, renderer.domElement);
  controls.zoomSpeed = 0.4;
  controls.panSpeed = 0.4;
  return controls;
}

/**
  * Load Nimrud model
  **/

function loadModel() {
  var loader = new THREE.OBJLoader();
  loader.load('./models/dragon.obj', function (object) {
    model = object;
    scene.add(object);
  });
}



/**
  * Render!
  **/

function render() {
  requestAnimationFrame(render);
  renderer.render(scene, camera);
  controls.update();
};

function readPoints(evt) {
  var files = evt.target.files;
  var file = files[0];
  var color = 0;
  if (file.name.includes("inside")) {
    color = 0x00ff00;
  }
  if (file.name.includes("outside")) {
    color = 0xff0000;
  }
  if (file.name.includes("singular")) {
    color = 0xFFD700;
  }
  var reader = new FileReader();
  reader.onload = function (event) {

    var text = reader.result;
    var points_text = text.split("\n");

    var pointsGeometry = new THREE.BufferGeometry();
    var point = new THREE.Vector3();
    var size = lineCount(text)
    const positions = new Float32Array( size * 3);
    var counter = 0;
    points_text.forEach(function (point_text) {
      point.x = parseFloat(point_text.split(" ")[0]);
      point.y = parseFloat(point_text.split(" ")[1]);
      point.z = parseFloat(point_text.split(" ")[2])
      point.toArray(positions, counter*3)
      counter++;
    });

    pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    var pointsMaterial = new THREE.PointsMaterial({ color: color, size: 0.003 });

    var points = new THREE.Points(pointsGeometry, pointsMaterial);

    scene.add(points);
  };
  reader.readAsText(file);
}

function lineCount( text ) {
  var nLines = 0;
  for( var i = 0, n = text.length;  i < n;  ++i ) {
      if( text[i] === '\n' ) {
          ++nLines;
      }
  }
  return nLines;
}

function readModel(evt) {
  var files = evt.target.files;
  var file = files[0];

  var reader = new FileReader();
  reader.onload = function (event) {
    //console.log(event.target.result);
    var text = reader.result;
    var loader = new THREE.OBJLoader();
    loader.parse(text, function (object) {

      object.traverse(function (child) {

        if (child.isMesh) {
          child.material.alphaTest = 0.5;
          child.material.transparent = true;
          child.material.opacity = 0.1;
          var wireframeGeomtry = new THREE.WireframeGeometry(child.geometry);
          var wireframeMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
          var wireframe = new THREE.LineSegments(wireframeGeomtry, wireframeMaterial);
          child.add(wireframe);

        }

      });

      //model = object;
      scene.add(object);
    });
  };
  reader.readAsText(file);
}


function showhide() {
  model.visible = !model.visible;
}

function wireframe() {

  model.traverse(function (child) {

    if (child.isMesh) {
      child.material.alphaTest = 0.5;
      child.material.transparent = true;
      child.material.opacity = 0.1;
      var wireframeGeomtry = new THREE.WireframeGeometry(child.geometry);
      var wireframeMaterial = new THREE.LineBasicMaterial({ color: 0x000000 });
      var wireframe = new THREE.LineSegments(wireframeGeomtry, wireframeMaterial);
      child.add(wireframe);

    }

  });

}
