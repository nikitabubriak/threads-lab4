import mpi.*;
import java.util.Date;

public class BmmMpj {
    // mpi-related values
    int taskid;
    int numtasks;
    int numworkers;

    // matrices
    double[] a;                // array must be one dimensional in mpiJava.
    double[] b;                // array must be one dimensional in mpiJava.
    double[] c;                // array must be one dimensional in mpiJava.

    // message component
    int averows;               // average #rows allocated to each rank
    int extra;                 // extra #rows allocated to some ranks
    int[] offset = new int[1]; // offset in row
    int[] rows = new int[1];   // the actual # rows allocated to each rank
    int source;
    int dest;
    int mtype;                 // message type (tagFromMaster or tagFromWorker )
    final static int master = 0;
    final static int tagFromMaster = 1;
    final static int tagFromWorker = 2;

    // print option
    boolean printOption;       // print out all array contents if true

    /**
     * Initializes matrices.
     * @param size the size of row/column for each matrix
     */
    private void init( int size ) {
        // Initialize matrices

        for ( int i = 0; i < size; i++ )
            for ( int j = 0; j < size; j++ )
                a[i * size + j] = i + j;       // a[i][j] = i + j;
        for ( int i = 0; i < size; i++ )
            for ( int j = 0; j < size; j++ )
                b[i * size + j] = i - j;       // b[i][j] = i - j;
        for ( int i = 0; i < size; i++ )
            for ( int j = 0; j < size; j++ )
                c[i * size + j] = 0;           // c[i][j] = 0
    }

    /**
     * Computes a multiplication for my allocated rows.
     * @param size the size of row/column for each matrix
     */
    private void compute( int size ) {

        for ( int k = 0; k < size; k++ )
            for ( int i = 0; i < rows[0]; i++ )
                for ( int j = 0; j < size; j++ ) {
                    // c[i][k] += a[i][j] * b[j][k]
                    c[i * size + k] += a[i * size + j] * b[j * size + k];
                }
    }

    /**
     * Prints out all elements of a given array.
     * @param array an array of doubles to print out
     */
    private void print(double[] array) {
        if ( taskid == 0 && printOption) {
            int size = ( int )Math.sqrt(array.length);
            for ( int i = 0; i < size; i++ )
                for ( int j = 0; j < size; j++ ) {
                    // System.out.println( array[i][j] );
                    System.out.println( "[" + i + "]"+ "[" + j + "] = "
                            + array[i * size + j] );
                }
        }
    }

    /**
     * Is the constructor that implements master-worker matrix transfers and
     * matrix multiplication.
     * @param option the size of row/column for each matrix
     * @param size   the option to print out all matrices ( print if true )
     */
    public BmmMpj(int size, boolean option ) throws MPIException {
        taskid = MPI.COMM_WORLD.Rank( );
        numtasks = MPI.COMM_WORLD.Size( );
        numworkers = numtasks - 1;

        a = new double[size * size];  // a = new double[size][size]
        b = new double[size * size];  // b = new double[size][size]
        c = new double[size * size];  // c = new double[size][size]

        printOption = option;

        if ( taskid == master ) {
            System.out.println("mpi has started with " + numtasks + " tasks");
            System.out.println( "size: " + size);
            // I'm a master. Initialize matrices.
            init( size );
            System.out.println( "array a:" );
            print( a );
            System.out.println( "array b:" );
            print( b );

            // Construct message components.
            averows = size / numworkers;
            extra = size % numworkers;
            offset[0] = 0;

            // Start timer.
            Date startTime = new Date( );

            // Transfer matrices to each worker.
            mtype = tagFromMaster;
            for (dest = 1; dest <= numworkers; dest++ ) {
                rows[0] = ( dest <= extra ) ? averows + 1 : averows;
                System.out.println( "sending " + rows[0] + " rows to task " + dest );

                MPI.COMM_WORLD.Send( offset, 0, 1, MPI.INT, dest, mtype );
                MPI.COMM_WORLD.Send( rows, 0, 1, MPI.INT, dest, mtype );
                MPI.COMM_WORLD.Send( a, offset[0] * size, rows[0] * size, MPI.DOUBLE, dest, mtype );
                MPI.COMM_WORLD.Send( b, 0, size * size, MPI.DOUBLE, dest, mtype );

                offset[0] += rows[0];
            }

            // Receive results from each worker.
            mtype = tagFromWorker;
            for (source = 1; source <= numworkers; source++ ) {
                MPI.COMM_WORLD.Recv( offset, 0, 1, MPI.INT, source, mtype );
                MPI.COMM_WORLD.Recv( rows, 0, 1, MPI.INT, source, mtype );
                MPI.COMM_WORLD.Recv( c, offset[0] * size, rows[0] * size, MPI.DOUBLE, source, mtype );
            }

            // Stop timer.
            Date endTime = new Date( );

            // Print out results
            System.out.println( "result c:" );
            print(c);

            System.out.println( "time elapsed = " + ( endTime.getTime( ) - startTime.getTime( ) ) + " ms" );
        }
        else {
            // I'm a worker. Receive matrices.
            mtype = tagFromMaster;
            MPI.COMM_WORLD.Recv( offset, 0, 1, MPI.INT, master, mtype );
            MPI.COMM_WORLD.Recv( rows, 0, 1, MPI.INT, master, mtype );
            MPI.COMM_WORLD.Recv( a, 0, rows[0] * size, MPI.DOUBLE, master, mtype );
            MPI.COMM_WORLD.Recv( b, 0, size * size, MPI.DOUBLE, master, mtype );

            // Perform matrix multiplication.
            compute(size);

            // Send results to the master.
            mtype = tagFromWorker;
            MPI.COMM_WORLD.Send( offset, 0, 1, MPI.INT, master, mtype );
            MPI.COMM_WORLD.Send( rows, 0, 1, MPI.INT, master, mtype );
            MPI.COMM_WORLD.Send( c, 0, rows[0] * size, MPI.DOUBLE, master, mtype );
        }

        System.out.println( "task[" + taskid + "] : multiplication completed" );
    }

    /**
     * @param args Receive the matrix size and the print option in args[0] and args[1]
     */
    public static void main( String[] args ) throws MPIException {
        //jar $MPJ_HOME$\lib\starter.jar BmmMpj -np 4 2 true

        // Start the MPI library.
        String appArgs[] = MPI.Init(args);

        int[] size = new int[1]; // square matrix order
        size[0] = Integer.parseInt(appArgs[0]);
        boolean[] option = new boolean[1]; // print option
        option[0] = Boolean.parseBoolean(appArgs[1]);

        // Compute matrix multiplication in both master and workers.
        new BmmMpj( size[0], option[0] );

        // Terminate the MPI library.
        MPI.Finalize();
    }
}